# RepoGraphForge: Clean Master Pipeline Guide (Kaggle L4x4)

This guide implements the complete, error-proof sequence from `DIAGNOSIS.md` and `FIX_PLAN.md`:

- **PART 0 (Phase 0 — CPU Ground-Truth Diagnostics):**
  - `0.1`: Dumps real Gemma 4 `chat_template` tokens, turn markers, and BOS behavior.
  - `0.2`: Verifies the exact tool-call format parsed by vLLM + ADK.
  - `0.3`: Measures trajectory token length percentiles (`[50, 90, 95, 99]`) and logs how many were previously destroyed by `MAX_LEN = 1536`.
  - `0.4`: Snapshots the current baseline adapter to `/kaggle/working/experiments/adapters/rft_v1_baseline` so it is never lost.
- **PART 1 (Setup & Single Canonical Formatter `chat_format.py`):**
  - Defines one shared `chat_format.py` (fixes **B1, B2, B4, B5, C1, C2**) with system prompt + 9 competition tool schemas, `MAX_LEN = 8192` drop-don't-truncate, single-BOS check, and mandatory Phase 1.2 pre-training decoded span printing.
- **PART 2 (Phase 4.1 — SFT Post-Training Only):**
  - Runs SFT with `MAX_LEN = 8192`, `lr = 1e-4`, `epochs = 2`, and prints the decoded trainable tokens for 3 samples before loading the 31B model.
- **PART 3 (Phase 4.2 — 3-Arm Holdout Evaluation & Corrected Audit Metric):**
  - Raises `max_output_tokens: 4096` (fixes **C3**), replaces the 1-file prohibition with the budget prompt (fixes **C4**), and removes the `or args == {}` bug from the audit table while logging `hit_max_output_tokens` (fixes **C5**).
- **PART 4 (Stretch Phase 6 — Gated RFT with Base-Model Judge):**
  - Disabled by default until SFT is verified (Phase 4.1); when enabled, uses the **base model** as judge (never `main_lora` grading itself) and enforces a hard minimum of `>= 50` accepted trajectories.

---

## 🧭 Master Cell Execution Map

```
┌────────────────────────────────────────────────────────────────────────┐
│            PART 0: PHASE 0 GROUND-TRUTH DIAGNOSTICS (CPU)              │
│  [Cell 0A] Phase 0.1 & 0.3: Dump Chat Template, BOS, & Token Lengths   │
│  [Cell 0B] Phase 0.4: Snapshot Current Baseline Adapter                │
│     (Phase 0.2 tool-schema diff = Cell 0C, runs in PART 1 — it needs   │
│      the ADK + Section 3 tools + chat_format.py to exist first.)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│            PART 1: SETUP & CANONICAL FORMATTER (chat_format.py)        │
│  [Cell 1] Section 1 from Ryan's Notebook (Install Dependencies)        │
│  [Cell 2] RepoGraphForge Patch: Fix Transformers & Disarm TorchAO      │
│  [Cell 3] Section 2 & 3 from Ryan's Notebook (Paths & Graph Utils)     │
│  [Cell 4] Write Canonical /kaggle/working/chat_format.py (Phases 1 & 3)│
│  [Cell 4B] Gate: assert the written chat_format.py is current          │
│  [Cell 0C] Phase 0.2: Diff real ADK tool schemas vs COMPETITION_TOOLS  │
│            *** BLOCKING — writes the receipt train_sft.py requires *** │
│  [Cell 0D] Phase 0.2 drill-down: run only if 0C flags a parameter      │
│  [Cell 0E] Phase 0.2: real graph-tool signatures + the tool registry   │
│  [Cell 0F] Phase 0.2 RESOLUTION: derive COMPETITION_TOOLS from harness │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│          PART 2: PHASE 4.1 SFT POST-TRAINING (NO RFT OVERWRITE)        │
│  [Cell 5] RepoGraphForge: GPU VRAM Flush & Health Check                │
│  [Cell 6] Write train_sft.py (chat_format + Phase 0.2 & 1.2 gates)     │
│  [Cell 7] Launch SFT Training (MAX_LEN=8192, lr=1e-4, 2 Epochs)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│          PART 3: PHASE 4.2 BENCHMARK, 3-ARM EVAL & AUDIT               │
│  [Cell 8] Sync Adapter, Set max_output_tokens=4096 & Budget Prompt     │
│  [Cell 9] Section 4 from Ryan's Notebook (Start vLLM Server)           │
│  [Cell 10] Section 5 from Ryan's Notebook (Run 14-Task Holdout Sweep)  │
│  [Cell 11] Corrected Holdout Audit Table (Fixes C5 + Logs Max-Tok Hits)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                PART 4: PACKAGING & LEADERBOARD SUBMISSION              │
│  [Cell 12] Section 6 from Ryan's Notebook (Package submission.zip)     │
│  [Cell 13] Direct Download Link / Submit to Leaderboard                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Cell-by-Cell Implementation

### PART 0: PHASE 0 GROUND-TRUTH DIAGNOSTICS (RUN ON CPU FIRST!)

#### Cell 0A: `[Phase 0.1 & 0.3]` Dump Real Chat Template, BOS Hygiene, & Trajectory Token Lengths
```python
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

MODEL_PATH = Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)

msgs = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "USR"},
    {"role": "assistant", "content": "THINK",
     "tool_calls": [{"id": "call_0", "type": "function",
                     "function": {"name": "read_file", "arguments": {"filepath": "a.py"}}}]},
    {"role": "tool", "name": "read_file", "tool_call_id": "call_0", "content": "OBS"},
    {"role": "assistant", "content": "DONE"},
]
rendered = tokenizer.apply_chat_template(msgs, tokenize=False)
print("=== PHASE 0.1: REAL GEMMA 4 CHAT TEMPLATE DUMP ===")
print(repr(rendered))
print("First 40 tokens:", tokenizer.tokenize(rendered)[:40])
print("BOS in template:", rendered.startswith(tokenizer.bos_token or "<bos>"))
print("add_special_tokens=True re-adds BOS:",
      tokenizer(rendered, add_special_tokens=True)["input_ids"][0] == tokenizer.bos_token_id
      and tokenizer(rendered, add_special_tokens=True)["input_ids"][1] == tokenizer.bos_token_id)
print("Special tokens:", tokenizer.all_special_tokens)

# Phase 0.3: Measure trajectory token lengths
traj_file = Path("/kaggle/working/data/trajectories/sft_combined.jsonl")
if not traj_file.exists():
    traj_file = Path("/kaggle/working/data/trajectories/sft_verified.jsonl")

if traj_file.exists():
    lens = []
    with open(traj_file) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                txt = ex.get("text") or json.dumps(ex.get("messages", []))
                lens.append(len(tokenizer(txt, add_special_tokens=False)["input_ids"]))
    print("\n=== PHASE 0.3: TRAJECTORY TOKEN LENGTH DISTRIBUTION ===")
    print("Percentiles [50, 90, 95, 99]:", np.percentile(lens, [50, 90, 95, 99]), "Max:", max(lens))
    print(f"Truncated at old MAX_LEN=1536: {sum(l > 1536 for l in lens)} / {len(lens)} ({sum(l > 1536 for l in lens)/max(1, len(lens))*100:.1f}%)")
    print(f"Overflow at new MAX_LEN=8192:  {sum(l > 8192 for l in lens)} / {len(lens)}")
```

---

#### Cell 0B: `[Phase 0.4]` Snapshot Current Baseline Adapter Before Touching Anything
```python
import shutil
from pathlib import Path

current_adapter = Path("/kaggle/working/submission/adapters/main_lora")
snapshot_dir = Path("/kaggle/working/experiments/adapters/rft_v1_baseline")

if current_adapter.exists() and list(current_adapter.glob("*.safetensors")):
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for f in current_adapter.glob("*"):
        if f.is_file():
            shutil.copy2(f, snapshot_dir / f.name)
    print(f"✓ Phase 0.4 Complete: Backed up existing adapter to {snapshot_dir}")
else:
    print("ℹ️ No existing adapter found at submission/adapters/main_lora.")
```

---

### PART 1: SETUP & CANONICAL FORMATTER (`chat_format.py`)

#### Cell 1: `[Section 1 from Ryan's Notebook]`
*Run unchanged from Ryan Holbrook's starter notebook.*  
Installs offline dependencies (`swegemma`, `adk_submission`, `adk_eval_core`) from the attached wheelhouse dataset.

---

#### Cell 2: `[RepoGraphForge Patch]` Fix Transformers 5.13.1 & Disarm TorchAO
```python
import os, sys
from pathlib import Path

print("=== APPLYING REPO-GRAPH-FORGE ENVIRONMENT SAFEGUARDS ===")

for p in [Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse"),
          Path("/kaggle/input/gemma-4-developer-agent-wheelhouse"),
          Path("/kaggle/input/competitions/gemma-4-developer-agent/wheels")]:
    if p.exists() and list(p.glob("*.whl")):
        !pip install -q --no-index --find-links={p} transformers safetensors
        print(f"✓ Installed transformers & safetensors from {p.name}")
        break

enc_file = Path("/usr/local/lib/python3.12/dist-packages/transformers/models/encoder_decoder/configuration_encoder_decoder.py")
if enc_file.exists():
    !sed -i 's/@auto_docstring.*//g' {enc_file}
    !sed -i 's/@strict.*//g' {enc_file}
    print("✓ Patched transformers encoder_decoder dataclass decorators.")

!pip uninstall -y torchao 2>/dev/null
peft_import = Path("/usr/local/lib/python3.12/dist-packages/peft/import_utils.py")
if peft_import.exists():
    !sed -i 's/raise ImportError.*incompatible version of torchao.*/return False/g' {peft_import}
    print("✓ Patched PEFT torchao check.")
```

---

#### Cell 3: `[Section 2 & Section 3 from Ryan's Notebook]`
*Run unchanged from Ryan Holbrook's starter notebook.*  
Sets up `DATA_DIR`, `WORKING_DIR`, `AGENT_DIR`, `GRAPH_DIR`, `EMBEDDINGS_DIR`, and graph exploration utilities.

---

#### Cell 4: `[Phase 1 & Phase 3]` Write Canonical `/kaggle/working/chat_format.py`
*(Note: `%%writefile` must be on Line 1 of this cell. Resolves B1, B2, B4, B5, C1, C2)*

```python
%%writefile /kaggle/working/chat_format.py
#!/usr/bin/env python3
"""
Phase 1 & Phase 3 (`FIX_PLAN.md`): Single Canonical Chat Formatter & Tokenizer (`src/chat_format.py`).

Resolves:
  - B1: Exact assistant-only loss masking INCLUSIVE of the terminating end-of-turn token (`<turn|>` / `<end_of_turn>`).
  - B2: `MAX_LEN = 8192` with strict drop-don't-truncate policy (never silently slices off `edit_file` / `submit_patch`).
  - B4: Single-BOS hygiene (`add_special_tokens=False` after `apply_chat_template`, asserting `bos_count <= 1`).
  - B5 & C1: Single shared serialization path across all scripts (`train_qlora.py`, `train_rft.py`, `teacher_rollouts.py`, `rft_sampler.py`, and notebooks).
  - C2: Injects the exact evaluation-time system prompt and 9 competition tool declarations (`COMPETITION_TOOLS`).
  - Phase 1.2: Mandatory pre-training assertions and decoded unmasked-token inspection on sample trajectories.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("chat_format")

# Bump on every change. The Kaggle notebook carries its own `%%writefile` copy of this
# file, so the two can drift; Cell 4B compares this string and refuses to proceed on a
# mismatch. A stale copy silently reinstates F1 and F2, which is how four runs were lost.
CHAT_FORMAT_VERSION = "2026-11-10.stage1d"

# --- F2 (silent preamble fallback) guards -------------------------------------------
# Phase 0.3 measured a MEDIAN trajectory length of 831 tokens against a tool-schema +
# system-prompt preamble of ~1,830 tokens. A median below the preamble is arithmetically
# impossible if the preamble is present, which proved the tool declarations were being
# dropped by a `try/except Exception: pass` on every single training example.
#
# STRICT_TOOLS=True makes that condition raise instead of passing silently.
STRICT_TOOLS: bool = True

# Substrings that MUST appear in rendered text when tools are declared. Drawn from the
# tool schemas themselves, so they fail if the template ignores an unsupported `tools=`.
# "node_id" was a probe until 2026-11-10. It was removed when F6 was corrected: the
# harness parameter is `node`, so probing for "node_id" would raise a FALSE F2 failure
# against correct schemas. Probes must be drawn from the CURRENT declarations.
TOOL_RENDER_PROBES = ("old_string", "nodes", "submit_patch")

# Populated whenever a render could not attach tool declarations; surfaced by
# `assert_tool_rendering_clean()` so a run cannot end without the failure being seen.
TOOL_RENDER_FAILURES: List[str] = []


# --- Masking-path accounting --------------------------------------------------------
# Phase 0.1 showed this template folds `<|tool_response>` blocks INSIDE the preceding
# model turn, so appending a tool message rewrites earlier tokens and the incremental
# prefix-diff path cannot hold. The span-marker fallback is therefore expected to run on
# essentially every example. We count both so the assumption is visible, not assumed.
MASK_PATH_COUNTS: Dict[str, int] = {"prefix_diff": 0, "span_marker": 0}


def _record_tool_render_failure(reason: str) -> None:
    TOOL_RENDER_FAILURES.append(reason)
    logger.error("TOOL SCHEMA RENDER FAILURE (failure mode F2): %s", reason)


def assert_tool_rendering_clean() -> None:
    """Call after dataset construction. Raises if any example lost its tool declarations."""
    if TOOL_RENDER_FAILURES:
        raise RuntimeError(
            f"{len(TOOL_RENDER_FAILURES)} tool-schema render failure(s); first: "
            f"{TOOL_RENDER_FAILURES[0]}"
        )
    logger.info("✓ Tool declarations present in every rendered example (F2 gate passed).")


# --- Phase 0.2 tool-schema verification receipt -------------------------------------
# A gate is only as good as its chokepoint. `assert_tool_rendering_clean()` above proves
# the declarations REACHED the training text; it cannot prove they are the RIGHT
# declarations. COMPETITION_TOOLS is a hand-written transcription of the harness tools,
# and training on a wrong parameter name produces a policy whose calls the sandbox
# rejects — a run that submits patches and scores zero, with no error at training time.
#
# Cell 0C performs the reconciliation and writes a receipt here. `train_sft.py` refuses
# to start without a receipt that is CURRENT (same chat_format version, same tool
# schemas) and CLEAN. There is no way to reach training without passing through it.
TOOL_SCHEMA_RECEIPT_PATH = Path("/kaggle/working/.tool_schema_verified.json")

# Outcomes Cell 0C may record:
#   "clean"        – real schemas found and every name + parameter matched
#   "mismatch"     – real schemas found and they differ; training is forbidden
#   "inconclusive" – no importable registry; only source-grep evidence available
VALID_OUTCOMES = ("clean", "mismatch", "inconclusive")


def tool_schema_fingerprint(tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """Stable hash of the tool schemas, so an edited parameter invalidates a receipt."""
    import hashlib

    tools = COMPETITION_TOOLS if tools is None else tools
    canonical = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def write_tool_schema_receipt(
    outcome: str,
    *,
    f2_ok: bool,
    f2_preamble_tokens: Optional[int] = None,
    discovery: str = "",
    mismatches: Optional[List[str]] = None,
    source_evidence: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> Path:
    """Called by Cell 0C. Records WHAT was verified, against WHICH schemas."""
    import datetime

    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {VALID_OUTCOMES}, got {outcome!r}")
    path = Path(path) if path is not None else TOOL_SCHEMA_RECEIPT_PATH
    receipt = {
        "chat_format_version": CHAT_FORMAT_VERSION,
        "tool_schema_fingerprint": tool_schema_fingerprint(),
        "tool_names": [
            (t.get("function", t) if isinstance(t, dict) else {}).get("name")
            for t in COMPETITION_TOOLS
        ],
        "outcome": outcome,
        "f2_ok": bool(f2_ok),
        "f2_preamble_tokens": f2_preamble_tokens,
        "discovery": discovery,
        "mismatches": list(mismatches or []),
        "source_evidence": source_evidence or {},
        "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    logger.info("✓ Tool-schema receipt written to %s (outcome=%s)", path, outcome)
    return path


def _gate_fail(reason: str, remedy: str) -> None:
    raise RuntimeError(
        "\n"
        + "=" * 72
        + "\nTOOL-SCHEMA GATE: TRAINING REFUSED\n"
        + "=" * 72
        + f"\n  Reason: {reason}\n\n  Remedy:\n{remedy}\n"
        + "=" * 72
        + "\n  Why this gate exists: training on wrong tool schemas costs ~3 GPU hours\n"
        "  and produces a run that converges, completes, submits patches, and scores\n"
        "  zero. There is no error at training time. This is the only place it can\n"
        "  be caught before the GPU spend.\n"
    )


def assert_tool_schema_verified(
    *,
    path: Optional[Path] = None,
    allow_inconclusive: Optional[bool] = None,
) -> Dict[str, Any]:
    """Hard gate. Call at the top of any training entrypoint. Returns the receipt.

    `allow_inconclusive` defaults to the TOOL_SCHEMA_ACK=1 environment variable, so
    proceeding on source-grep evidence alone is a deliberate, recorded act.
    """
    import os

    path = Path(path) if path is not None else TOOL_SCHEMA_RECEIPT_PATH
    if allow_inconclusive is None:
        allow_inconclusive = os.environ.get("TOOL_SCHEMA_ACK") == "1"

    if not path.exists():
        _gate_fail(
            f"no verification receipt at {path}",
            "    Run Cell 0C (Phase 0.2 tool-schema reconciliation). It writes the\n"
            "    receipt as its last step.",
        )

    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _gate_fail(f"receipt at {path} is unreadable ({e!r})", "    Delete it and re-run Cell 0C.")

    got_ver = receipt.get("chat_format_version")
    if got_ver != CHAT_FORMAT_VERSION:
        _gate_fail(
            f"receipt was written against chat_format {got_ver!r}, "
            f"but this is {CHAT_FORMAT_VERSION!r}",
            "    chat_format.py changed since verification. Re-run Cell 0C.",
        )

    got_fp = receipt.get("tool_schema_fingerprint")
    cur_fp = tool_schema_fingerprint()
    if got_fp != cur_fp:
        _gate_fail(
            f"COMPETITION_TOOLS changed since verification "
            f"(receipt {got_fp}, current {cur_fp})",
            "    The tool schemas were edited after Cell 0C ran. Re-run Cell 0C so the\n"
            "    verification applies to the schemas you are about to train on.",
        )

    if not receipt.get("f2_ok"):
        _gate_fail(
            "the F2 probe failed: tool declarations do not survive into rendered text",
            "    The chat template is dropping `tools=`. Fix rendering before training;\n"
            "    a model trained without declarations never learns the tool surface.",
        )

    outcome = receipt.get("outcome")
    if outcome == "mismatch":
        ms = receipt.get("mismatches") or []
        detail = "".join(f"      {m}\n" for m in ms[:10]) or "      (none recorded)\n"
        _gate_fail(
            "Cell 0C found tool-schema mismatches:\n" + detail.rstrip(),
            "    Reconcile COMPETITION_TOOLS with the harness. Prefer importing the real\n"
            "    schemas over editing the copy — a copy drifts, an import cannot.",
        )

    if outcome == "inconclusive" and not allow_inconclusive:
        _gate_fail(
            "Cell 0C could not locate importable harness schemas; only source-grep "
            "evidence is available",
            "    Read the source evidence Cell 0C printed. If every tool name and\n"
            "    parameter name was found in the harness source and you accept that as\n"
            "    sufficient, re-run training with TOOL_SCHEMA_ACK=1 set:\n"
            "        os.environ['TOOL_SCHEMA_ACK'] = '1'\n"
            "    Setting it is a recorded decision to train on unconfirmed schemas.",
        )

    logger.info(
        "✓ Tool-schema gate passed (outcome=%s, f2_preamble=%s tokens, discovery=%s)",
        outcome,
        receipt.get("f2_preamble_tokens"),
        receipt.get("discovery") or "n/a",
    )
    if outcome == "inconclusive":
        logger.warning(
            "⚠ Training on UNCONFIRMED tool schemas (TOOL_SCHEMA_ACK=1). "
            "Record this in the run log — it is a limitation of the resulting model."
        )
    return receipt


# Phase 3.4 Budget-Based System Prompt (replaces the brittle 1-file/turn-3 prohibitions)
DEFAULT_SYSTEM_PROMPT = """You are an autonomous software engineering agent equipped with a pre-computed repository code graph.

You have up to 15 tool calls. Spend them with discipline:
1. LOCALIZE VIA GRAPH FIRST:
   - Start with `get_code_subgraph`, `get_code_neighbors`, or `search_similar_code` on the symbol or symptom named in the issue/traceback.
   - Symbols are dotted paths or suffixes: 'pkg.module.Class.method', 'Class.method', or 'parse_request'.
   - `get_code_subgraph(nodes=[...])` takes a LIST of symbols. `get_code_neighbors(node="...")` takes a single symbol.
   - If `get_code_subgraph` returns 0 nodes because the exact symbol is unknown, call `search_similar_code(query="...")` to find the exact symbol and file path.
2. READ TARGETED SPANS (AT MOST 3 FILES):
   - Read at most 3 files using `read_file(filepath, start_line, end_line)` on the 30-80 lines around the target symbol.
   - Prefer inspecting the callee/helper function that the subgraph points to over paging through large caller files.
3. APPLY EXACT EDITS (`edit_file`):
   - Call `edit_file(filepath, old_string, new_string)` using the exact `old_string` copied verbatim from your `read_file` output.
   - If `edit_file` returns `FileEditError`, read the exact line range first and retry `edit_file`.
4. TEST & SUBMIT (`submit_patch`):
   - If you have tool calls remaining after `edit_file` succeeds, run the relevant unit test with `run_command` (e.g. `pytest tests/test_... -k ...`) to verify your fix.
   - Always finish with `submit_patch()` once a valid edit is applied. It takes NO arguments — it submits the current working-tree diff. Never call `submit_patch` before `edit_file` has succeeded."""

# --- F6: the tool interface -----------------------------------------------------------
# These declarations were ORIGINALLY hand-transcribed and diverged from the harness in
# five of nine tools (Phase 0.2, 2026-11-10). The transcription was internally consistent
# — valid schema, rendered fine, learned faithfully — so nothing inside training could
# detect it. The values below are taken from `inspect.signature` on the live callables in
# `swegemma.tools` and are recorded here only as a FALLBACK.
#
# The authoritative path is `real_tools.json`, written by Cell 0F from the harness itself.
# A copy drifts; a derivation cannot. Prefer the derivation whenever it is present.
#
# `ctx` is injected by the harness and never supplied by the model, so it is not declared.
_FALLBACK_COMPETITION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_code_subgraph",
            "description": "Extract the relationship graph (nodes and connecting edges) for a focal set of code elements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of dotted symbol paths or suffix names (e.g. ['pkg.mod.Class.method', 'helper_fn']) to include in the induced subgraph.",
                    },
                },
                "required": ["nodes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_code_neighbors",
            "description": "Discover structural code graph neighbors (callers and callees) for a given code node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": {
                        "type": "string",
                        "description": "Dotted symbol name or suffix of the function, method, or class node (e.g. 'pkg.module.Class.method', 'Class.method', or 'parse_request').",
                    },
                    "edge_type": {
                        "type": "string",
                        "description": "Optional edge relation filter (e.g. 'calls'; case-insensitive, default None for all edges).",
                    },
                    "max_neighbors": {
                        "type": "integer",
                        "description": "Maximum number of neighbors to return (default 50).",
                    },
                },
                "required": ["node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_similar_code",
            "description": "Find code elements semantically similar to a query string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Symbol name, error message, or code concept to search."},
                    "k": {"type": "integer", "description": "Number of results to return (default 10)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository, optionally restricted to a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Repository-relative path to the file."},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed, optional)."},
                    "end_line": {"type": "integer", "description": "Last line to read (inclusive, optional)."},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Repository-relative path to the file."},
                    "old_string": {"type": "string", "description": "Exact existing text to replace, copied verbatim from read_file output."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "allow_multiple": {"type": "boolean", "description": "Replace every occurrence rather than requiring a unique match (default false)."},
                },
                "required": ["filepath", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the repository, creating or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Repository-relative path to the file."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the repository sandbox and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute, e.g. 'pytest tests/test_x.py -k name'."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Report the current workspace status, including files modified so far.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": "Finalize and submit the current working-tree diff as the answer. Takes no arguments.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Written by Cell 0F via inspect.signature over `swegemma.tools`. When present it wins.
REAL_TOOLS_PATH = Path("/kaggle/working/real_tools.json")
TOOLS_SOURCE = "fallback (hand-written, verified against Phase 0.2 signatures)"

# A derived artefact needs its own version gate, for the same reason this module has one.
# The first Cell 0F inferred JSON types by substring-matching the annotation; "list[str]"
# contains "str", so it typed the list-valued `nodes` parameter as a string — F6 reborn
# inside the fix for F6. Any real_tools.json written before this stamp is refused.
REQUIRED_REAL_TOOLS_GENERATOR = "2026-11-10.0F-v2"


def _load_competition_tools() -> List[Dict[str, Any]]:
    global TOOLS_SOURCE
    try:
        if not REAL_TOOLS_PATH.exists():
            return _FALLBACK_COMPETITION_TOOLS
        data = json.loads(REAL_TOOLS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            logger.warning(
                "%s is unversioned (written by Cell 0F v1, which mistyped list[str] as "
                "string). REFUSING it; using fallback schemas. Re-run Cell 0F.",
                REAL_TOOLS_PATH,
            )
            return _FALLBACK_COMPETITION_TOOLS
        gen = (data or {}).get("generator_version")
        tools = (data or {}).get("tools")
        if gen != REQUIRED_REAL_TOOLS_GENERATOR:
            logger.warning(
                "%s was written by generator %r but %r is required. REFUSING it; using "
                "fallback schemas. Re-run Cell 0F.",
                REAL_TOOLS_PATH, gen, REQUIRED_REAL_TOOLS_GENERATOR,
            )
            return _FALLBACK_COMPETITION_TOOLS
        if not (isinstance(tools, list) and tools and all(isinstance(d, dict) for d in tools)):
            logger.warning("%s is malformed; using fallback schemas.", REAL_TOOLS_PATH)
            return _FALLBACK_COMPETITION_TOOLS
        TOOLS_SOURCE = f"{REAL_TOOLS_PATH} (generator {gen})"
        logger.info("✓ Tool schemas derived from the harness: %s", REAL_TOOLS_PATH)
        return tools
    except Exception as e:  # noqa: BLE001 - never let schema loading kill the import
        logger.warning("Could not read %s (%r); using fallback schemas.", REAL_TOOLS_PATH, e)
    return _FALLBACK_COMPETITION_TOOLS


COMPETITION_TOOLS: List[Dict[str, Any]] = _load_competition_tools()


def sanitize_user_prompt(content: str) -> str:
    """
    Removes synthetic training headers ('### Task ID:', '### Anchor Node:') from user prompts
    so the fine-tuned policy never relies on leaked node IDs at evaluation time.
    """
    if not content:
        return ""
    cleaned = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("### Task ID:") or stripped.startswith("### Anchor Node:"):
            continue
        if stripped == "### Problem Statement:":
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def normalize_messages(
    raw_messages: List[Dict[str, Any]],
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """
    Standardizes a trajectory message list into the exact format required by Gemma 4's
    `chat_template.jinja`:
      - Prepends the system prompt if not already present (fixes C2).
      - Ensures `tool_calls[i].function.arguments` is a Python `dict` (mapping).
      - Attaches `name` and `tool_call_id` to every `role == 'tool'` message.
      - Strips leaked synthetic headers from the first user prompt.
    """
    standardized: List[Dict[str, Any]] = []
    has_system = any(m.get("role") == "system" for m in raw_messages)
    if system_prompt and not has_system:
        standardized.append({"role": "system", "content": system_prompt})

    last_tool_name = "get_code_subgraph"
    last_call_id = "call_0"
    seen_first_user = False

    for idx_m, m in enumerate(raw_messages):
        role = m.get("role", "user")
        content = m.get("content") or ""

        if role == "system":
            standardized.append({"role": "system", "content": content})

        elif role in ("assistant", "model"):
            msg: Dict[str, Any] = {"role": "assistant", "content": content}
            tcs = m.get("tool_calls", [])
            if tcs and isinstance(tcs, list):
                norm_tcs = []
                for idx, tc in enumerate(tcs):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                    name = fn.get("name") or tc.get("name") or "get_code_subgraph"
                    args = fn.get("arguments") if "arguments" in fn else tc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"query": args}
                    if not isinstance(args, dict):
                        args = {"value": args}
                    cid = tc.get("id") or f"call_{idx_m}_{idx}"
                    last_tool_name = name
                    last_call_id = cid
                    norm_tcs.append({
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
                if norm_tcs:
                    msg["tool_calls"] = norm_tcs
            standardized.append(msg)

        elif role == "tool":
            standardized.append({
                "role": "tool",
                "name": m.get("name") or last_tool_name,
                "tool_call_id": m.get("tool_call_id") or last_call_id,
                "content": content,
            })

        else:
            if not seen_first_user:
                content = sanitize_user_prompt(content)
                seen_first_user = True
            standardized.append({"role": "user", "content": content})

    return standardized


def render(
    messages: List[Dict[str, Any]],
    tokenizer: Any,
    tools: Optional[List[Dict[str, Any]]] = COMPETITION_TOOLS,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    add_generation_prompt: bool = False,
) -> str:
    """
    Single canonical rendering function (`FIX_PLAN.md` Phase 3.1).
    Formats messages with the exact system prompt and tool declarations.
    """
    norm_msgs = normalize_messages(messages, system_prompt=system_prompt)

    if not tools:
        return tokenizer.apply_chat_template(
            norm_msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    try:
        rendered = tokenizer.apply_chat_template(
            norm_msgs,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as e:
        # F2 FIX: FAIL CLOSED. The previous `except Exception: pass` silently dropped the
        # tool declarations from EVERY training sequence while reporting success, so the
        # policy never saw the schemas the eval harness sends. Never swallow this again.
        _record_tool_render_failure(f"apply_chat_template(tools=...) raised: {e!r}")
        if STRICT_TOOLS:
            raise RuntimeError(
                "Tool schemas could not be rendered into the chat template. "
                "Training without them reproduces failure F2 (train/eval preamble mismatch). "
                "Fix the template call, or set chat_format.STRICT_TOOLS = False to proceed "
                "deliberately and without tool declarations."
            ) from e
        return tokenizer.apply_chat_template(
            norm_msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # The call may succeed while silently ignoring an unsupported `tools` kwarg, which is
    # exactly how F2 went undetected. Verify the declarations are actually in the text.
    missing = [p for p in TOOL_RENDER_PROBES if p not in rendered]
    if missing:
        _record_tool_render_failure(
            f"apply_chat_template accepted tools= but omitted {missing} from the output"
        )
        if STRICT_TOOLS:
            raise RuntimeError(
                f"Tool schemas absent from rendered text (missing probes: {missing}). "
                "The template accepted `tools=` but did not emit the declarations. "
                "This is failure F2; training would silently mismatch eval-time context."
            )
    return rendered


def _find_eot_token_id(tokenizer: Any) -> Optional[int]:
    """Detects the real end-of-turn token ID used by the loaded Gemma tokenizer."""
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for candidate in ("<turn|>", "<end_of_turn>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(candidate)
        if isinstance(tid, int) and tid >= 0 and tid != unk_id:
            return tid
    return getattr(tokenizer, "eos_token_id", None)


def build_example(
    raw_messages: List[Dict[str, Any]],
    tokenizer: Any,
    max_len: int = 8192,
    tools: Optional[List[Dict[str, Any]]] = COMPETITION_TOOLS,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> Optional[Dict[str, List[int]]]:
    """
    Implements `FIX_PLAN.md` Phase 1.1–1.4:
      1. Incremental prefix-diff tokenization across `norm_msgs` so every assistant span
         is unmasked INCLUSIVE of its terminating end-of-turn token (`<turn|>` / `<end_of_turn>`).
      2. Falls back to exact token boundary masking if the Jinja template folds `tool` responses
         inside the preceding `model` turn block.
      3. Uses `add_special_tokens=False` on rendered text so `<bos>` is NEVER duplicated (fixes B4).
      4. Returns `None` if `len(input_ids) > max_len` (strict drop-don't-truncate policy, fixes B2).
    """
    norm_msgs = normalize_messages(raw_messages, system_prompt=system_prompt)

    # First try incremental turn-by-turn prefix diff (FIX_PLAN.md Phase 1.1)
    input_ids: List[int] = []
    labels: List[int] = []
    prefix_ok = True

    try:
        prev_ids: List[int] = []
        for i, m in enumerate(norm_msgs):
            sub_msgs = norm_msgs[: i + 1]
            # F2: no silent tools-dropping fallback here either. `render` fails closed.
            rendered_upto = render(
                sub_msgs,
                tokenizer,
                tools=tools,
                system_prompt=None,  # already normalized; do not re-prepend
            )

            # B4 fix: template already emits <bos>, so tokenize with add_special_tokens=False!
            upto_ids = list(tokenizer(rendered_upto, add_special_tokens=False)["input_ids"])
            if len(upto_ids) < len(prev_ids) or upto_ids[: len(prev_ids)] != prev_ids:
                # Template modified earlier tokens (e.g. inline tool response folding in Gemma 4)
                prefix_ok = False
                break

            span = upto_ids[len(prev_ids) :]
            input_ids.extend(span)
            if m["role"] in ("assistant", "model"):
                labels.extend(_mask_rendered_token_ids(span, tokenizer))
            else:
                labels.extend([-100] * len(span))
            prev_ids = upto_ids
    except Exception:
        prefix_ok = False

    if not prefix_ok or not input_ids or all(l == -100 for l in labels):
        # Exact span mask on the full rendered string (handles inline <|tool_response> blocks)
        MASK_PATH_COUNTS["span_marker"] += 1
        rendered_full = render(raw_messages, tokenizer, tools=tools, system_prompt=system_prompt)
        input_ids = list(tokenizer(rendered_full, add_special_tokens=False)["input_ids"])
        labels = _mask_rendered_token_ids(input_ids, tokenizer)
    else:
        MASK_PATH_COUNTS["prefix_diff"] += 1

    # Ensure single BOS at index 0 if tokenizer defines bos_token_id
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        # Remove any accidental duplicate leading BOS tokens (fixes B4)
        while len(input_ids) >= 2 and input_ids[0] == bos_id and input_ids[1] == bos_id:
            input_ids.pop(0)
            labels.pop(0)
        if input_ids and input_ids[0] == bos_id:
            labels[0] = -100

    # B2 fix: Drop (do NOT truncate) any trajectory that exceeds max_len!
    if len(input_ids) > max_len:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _mask_rendered_token_ids(input_ids: List[int], tokenizer: Any) -> List[int]:
    """
    Token-level span masker supporting both Gemma 4 (`<|turn>model ... <turn|>`)
    and Gemma 2/3 (`<start_of_turn>model ... <end_of_turn>`), masking inline
    `<|tool_response> ... <tool_response|>` observations while keeping all assistant
    turns and closing end-of-turn tokens (`<turn|>` / `<end_of_turn>`) trainable!
    """
    labels = [-100] * len(input_ids)
    unk_id = getattr(tokenizer, "unk_token_id", -1)

    def _tid(t: str) -> int:
        tid = tokenizer.convert_tokens_to_ids(t)
        return tid if (isinstance(tid, int) and tid != unk_id) else -999

    turn_open_ids = {x for x in (_tid("<|turn>"), _tid("<start_of_turn>")) if x >= 0}
    turn_close_ids = {x for x in (_tid("<turn|>"), _tid("<end_of_turn>")) if x >= 0}
    tool_resp_open = _tid("<|tool_response>")
    tool_resp_close = _tid("<tool_response|>")
    system_role_ids = tokenizer.encode("system", add_special_tokens=False)
    user_role_ids = tokenizer.encode("user", add_special_tokens=False)
    model_role_ids = tokenizer.encode("model", add_special_tokens=False)
    nl_ids = set(tokenizer.encode("\n", add_special_tokens=False))
    bos_id = getattr(tokenizer, "bos_token_id", None)

    in_system = False
    in_user = False
    in_tool_response = False

    i = 0
    n = len(input_ids)

    while i < n:
        if input_ids[i] in turn_open_ids:
            if system_role_ids and input_ids[i + 1 : i + 1 + len(system_role_ids)] == system_role_ids:
                in_system = True
                i += 1 + len(system_role_ids)
                continue
            elif user_role_ids and input_ids[i + 1 : i + 1 + len(user_role_ids)] == user_role_ids:
                in_user = True
                i += 1 + len(user_role_ids)
                continue
            elif model_role_ids and input_ids[i + 1 : i + 1 + len(model_role_ids)] == model_role_ids:
                i += 1 + len(model_role_ids)
                if i < n and input_ids[i] in nl_ids:
                    i += 1
                continue

        if input_ids[i] in turn_close_ids:
            if in_system:
                in_system = False
                i += 1
                continue
            if in_user:
                in_user = False
                i += 1
                continue
            # End-of-turn token on model generation carries gradient (B1 fix!)
            labels[i] = input_ids[i]
            i += 1
            continue

        if input_ids[i] == tool_resp_open:
            in_tool_response = True
            i += 1
            continue

        if input_ids[i] == tool_resp_close:
            in_tool_response = False
            i += 1
            continue

        if not in_system and not in_user and not in_tool_response:
            if bos_id is not None and input_ids[i] == bos_id:
                pass
            elif input_ids[i] in nl_ids and (i == 0 or input_ids[i - 1] in turn_close_ids):
                pass
            else:
                labels[i] = input_ids[i]

        i += 1

    return labels

def verify_and_print_samples(
    tokenized_samples: List[Dict[str, List[int]]],
    tokenizer: Any,
    num_samples_to_print: int = 3,
) -> None:
    """
    Implements `FIX_PLAN.md` Phase 1.2 & 1.4 mandatory pre-training verification gate:
      - Asserts every example has both masked (-100) and unmasked (!= -100) tokens.
      - Asserts single BOS token (`input_ids.count(bos_id) <= 1`).
      - Asserts the end-of-turn token ID is present in the unmasked labels.
      - Prints the decoded unmasked tokens for `num_samples_to_print` examples so you
        can visually verify that ONLY assistant turns + `<turn|>` carry gradient.
    """
    if not tokenized_samples:
        raise ValueError("Tokenized dataset is empty!")

    eot_id = _find_eot_token_id(tokenizer)
    bos_id = getattr(tokenizer, "bos_token_id", None)

    print("=" * 80)
    print("=== PHASE 1.2 PRE-TRAINING MASKING & EOT VERIFICATION GATE ===")
    print("=" * 80)

    # Text that must NEVER carry gradient. If any of these appear in the supervised
    # tokens, the model is being trained to hallucinate its own environment observations
    # instead of calling tools — the second half of failure F1.
    OBSERVATION_MARKERS = (
        "<|tool_response>",      # native folded tool-response block
        "File: ",                # read_file observation header
        '"status": "ok"',        # edit_file / submit_patch observation payload
        '"replacements"',        # edit_file observation payload
        "### Problem",           # user-turn task text
    )

    n_checked = 0
    for idx, ex in enumerate(tokenized_samples[: max(5, num_samples_to_print)]):
        ids = ex["input_ids"]
        lab = ex["labels"]
        unmasked_ids = [i for i, l in zip(ids, lab) if l != -100]
        n_checked += 1

        assert any(l != -100 for l in lab), f"Sample {idx}: all labels are masked (-100)!"
        assert any(l == -100 for l in lab), f"Sample {idx}: nothing masked — user/system turns are trainable!"

        # Supervised fraction. The old `any(l == -100)` check was satisfied by the system
        # prompt alone, so it passed even when every observation was trainable. A healthy
        # agent trajectory supervises roughly 5-35% of its tokens.
        frac = len(unmasked_ids) / max(1, len(ids))
        assert 0.03 < frac < 0.45, (
            f"Sample {idx}: supervised fraction {frac:.1%} out of range "
            f"({len(unmasked_ids)}/{len(ids)} tokens). "
            f"Too high => user/observation turns are trainable; too low => assistant spans are masked."
        )

        # Observation-leakage check.
        decoded_unmasked = tokenizer.decode(unmasked_ids, skip_special_tokens=False)
        for marker in OBSERVATION_MARKERS:
            assert marker not in decoded_unmasked, (
                f"Sample {idx}: observation text {marker!r} leaked into trainable labels — "
                f"the model is being taught to generate its own tool outputs."
            )

        if bos_id is not None and bos_id in ids:
            bos_count = ids.count(bos_id)
            assert bos_count == 1, f"Sample {idx}: Double-BOS detected (bos_count={bos_count})!"
        if eot_id is not None and eot_id in ids:
            assert eot_id in unmasked_ids, (
                f"Sample {idx}: End-of-turn token (id={eot_id}) is NOT trainable in labels! "
                f"This is failure F1: the policy will never learn to terminate a turn."
            )

        if idx < num_samples_to_print:
            print(f"\n--- Sample {idx} Unmasked Trainable Tokens ({len(unmasked_ids)}/{len(ids)} = {frac:.1%}) ---")
            print(decoded_unmasked[:900] + (" ... [truncated]" if len(decoded_unmasked) > 900 else ""))

    print(f"\n[mask] paths used: {MASK_PATH_COUNTS}")
    print(f"[mask] samples checked: {n_checked}")
    assert_tool_rendering_clean()
    print("\n✓ Phase 1.2 Gate PASSED: assistant-only spans, EOT supervised, no observation leakage.")
    print("  READ THE DECODED TEXT ABOVE. It must be assistant turns only — nothing else.")
    print("=" * 80)
```

---

#### Cell 4B: `[Stage 1 Gate]` Verify the Written `chat_format.py` Is Current

*Cell 4's body is generated from `src/chat_format.py`. This cell fails loudly if an old
copy is in play — a stale formatter silently reintroduces F1/F2.*

```python
import sys, importlib
sys.path.insert(0, "/kaggle/working")
import chat_format
importlib.reload(chat_format)

EXPECTED_VERSION = "2026-11-10.stage1d"
REQUIRED = ["CHAT_FORMAT_VERSION", "STRICT_TOOLS", "TOOL_RENDER_PROBES",
            "MASK_PATH_COUNTS", "assert_tool_rendering_clean",
            "build_example", "verify_and_print_samples",
            "write_tool_schema_receipt", "assert_tool_schema_verified",
            "tool_schema_fingerprint", "TOOLS_SOURCE", "REAL_TOOLS_PATH"]

missing = [n for n in REQUIRED if not hasattr(chat_format, n)]
got = getattr(chat_format, "CHAT_FORMAT_VERSION", "<pre-versioning>")

if missing or got != EXPECTED_VERSION:
    print("=" * 72)
    print("STALE /kaggle/working/chat_format.py")
    print("=" * 72)
    print(f"  expected version : {EXPECTED_VERSION}")
    print(f"  loaded version   : {got}")
    if missing:
        print(f"  missing symbols  : {missing}")
    print(f"  file written at  : {getattr(chat_format, '__file__', '?')}")
    print()
    print("  CAUSE: this notebook's Cell 4 is an older copy. The repo markdown is a")
    print("         guide; the cell you actually execute lives in this notebook and")
    print("         does not update when the repo does.")
    print()
    print("  FIX:   1. Open src/chat_format.py in the repo.")
    print("         2. Replace EVERYTHING in Cell 4 below the %%writefile line with it.")
    print("         3. Run Cell 4, then re-run this cell.")
    print()
    print("  Do NOT skip this. A stale formatter silently reinstates F1 (stop token")
    print("  unsupervised) and F2 (tool schemas dropped) with no error at training time.")
    print("=" * 72)
    raise SystemExit(1)

assert chat_format.STRICT_TOOLS is True, "STRICT_TOOLS is off; tool-render failures pass silently (F2)."
print(f"\u2713 chat_format.py is current (version {got}); STRICT_TOOLS enabled.")
print(f"  loaded from      : {chat_format.__file__}")
print(f"  tool probes      : {chat_format.TOOL_RENDER_PROBES}")
print(f"  masking paths    : {chat_format.MASK_PATH_COUNTS}")
print(f"  tool schemas from: {chat_format.TOOLS_SOURCE}")
_names = [t["function"]["name"] for t in chat_format.COMPETITION_TOOLS]
print(f"  tools declared   : {len(_names)} {_names}")
import json as _json
assert "node_id" not in _json.dumps(chat_format.COMPETITION_TOOLS), \
    "F6 regression: node_id is back in the schemas. The harness parameter is `node`/`nodes`."
```

---

#### Cell 0C: `[Phase 0.2]` Diff Real ADK Tool Schemas Against `COMPETITION_TOOLS`

> **Why this cell exists.** `src/chat_format.py:COMPETITION_TOOLS` is a hand-written
> transcription of the competition's 9 tools. Nothing has ever checked it against the
> real harness. If one parameter name differs, training teaches the policy to emit calls
> the sandbox rejects — which is exactly the signature of a run that submits patches and
> scores zero.

> **Prerequisites:** Cell 1 (installs `swegemma` / `adk_eval_core` / `adk_submission`),
> Cell 3, Cell 4, Cell 4B. Running it before those is why it sat unexecuted.

> **v1 post-mortem (2026-11-10).** The first version guessed six global names
> (`TOOLS`, `tools`, `TOOL_SCHEMAS`, …), found none, and reported
> *"Could not locate the tool definitions automatically."* That was a bad assumption:
> the harness tools are not notebook globals — they live inside the installed ADK
> packages. Worse, the F2 render probe was nested inside the `else`, so a discovery
> miss silently skipped the one check that needed no discovery at all. v2 searches
> globals **structurally** (any list whose head looks like a tool), walks the ADK
> packages, falls back to grepping the harness **source on disk** for our 9 tool names,
> and always runs the F2 probe.

```python
# =============================================================================
# Phase 0.2 — Tool schema reconciliation.  Version 3 (writes the training gate).
#
# v1 probed six guessed global names, found none, and then skipped the F2 render
# probe because it was nested inside the `else`. v2 fixed discovery and unnested
# the probe. v3 writes a verification RECEIPT that `train_sft.py` refuses to
# start without — so a wrong tool schema can no longer reach the GPU.
# =============================================================================
import json, inspect, sys, importlib, pkgutil, re
from pathlib import Path
sys.path.insert(0, "/kaggle/working")
from chat_format import (
    COMPETITION_TOOLS, CHAT_FORMAT_VERSION,
    write_tool_schema_receipt, TOOL_SCHEMA_RECEIPT_PATH,
)

CELL_0C_VERSION = "v3 (2026-11-10) — writes the training gate receipt"
print(f"CELL 0C {CELL_0C_VERSION}")
print(f"chat_format version: {CHAT_FORMAT_VERSION}")
OUR_NAMES = []
for _t in COMPETITION_TOOLS:
    _f = _t.get("function", _t) if isinstance(_t, dict) else {}
    if _f.get("name"):
        OUR_NAMES.append(_f["name"])
print(f"We declare {len(OUR_NAMES)} tools: {OUR_NAMES}\n")

# ---------------------------------------------------------------- A. globals
def _looks_like_tool_list(obj):
    if not isinstance(obj, (list, tuple)) or not obj:
        return False
    head = obj[0]
    if isinstance(head, dict) and ("name" in head or "function" in head):
        return True
    if callable(head) and getattr(head, "__name__", "").islower():
        return True
    return any(hasattr(head, a) for a in ("declaration", "_declaration", "name", "func"))

real_tools, found_via = None, None
for _k, _v in list(globals().items()):
    if _k.startswith("_") or _k == "COMPETITION_TOOLS":
        continue
    if _looks_like_tool_list(_v):
        real_tools, found_via = list(_v), f"global `{_k}`"
        break

# ------------------------------------------------- B. installed ADK packages
PKGS = ("swegemma", "adk_eval_core", "adk_submission", "adk")
pkg_paths = {}
for _p in PKGS:
    try:
        _m = importlib.import_module(_p)
        pkg_paths[_p] = Path(getattr(_m, "__file__", "") or "").parent
    except Exception:
        pkg_paths[_p] = None

if real_tools is None:
    for _p, _root in pkg_paths.items():
        if _root is None:
            continue
        try:
            _mod = importlib.import_module(_p)
        except Exception:
            continue
        _subs = [_mod]
        try:
            for _mi in pkgutil.walk_packages([str(_root)], prefix=_p + "."):
                try:
                    _subs.append(importlib.import_module(_mi.name))
                except Exception:
                    pass
        except Exception:
            pass
        for _sm in _subs:
            for _k, _v in vars(_sm).items():
                if _k.startswith("_"):
                    continue
                if _looks_like_tool_list(_v):
                    real_tools, found_via = list(_v), f"{_sm.__name__}.{_k}"
                    break
                if isinstance(_v, dict) and _v and all(callable(x) for x in _v.values()):
                    if set(_v) & set(OUR_NAMES) or len(_v) >= 5:
                        real_tools = list(_v.values())
                        found_via = f"{_sm.__name__}.{_k} (name->callable registry)"
                        break
            if real_tools:
                break
        if real_tools:
            break

if real_tools is not None:
    print(f"✓ Found tool definitions via {found_via} ({len(real_tools)} tools)\n")

# ------------------------------------------- C. filesystem evidence (always)
# Even without an importable registry, the harness SOURCE is on disk. If a tool
# name we train on never appears anywhere in it, that name is wrong — which is
# sufficient on its own to explain patches that submit and score zero.
print("=" * 72)
print("SOURCE-LEVEL EVIDENCE: do our tool names exist in the harness at all?")
print("=" * 72)
CRIT_PARAMS = ["old_string", "new_string", "filepath", "node_id", "patch", "command"]
corpus, scanned = {}, 0
for _p, _root in pkg_paths.items():
    if _root is None or not _root.exists():
        print(f"  (package {_p} not importable — skipped)")
        continue
    for _f in _root.rglob("*.py"):
        try:
            corpus[str(_f)] = _f.read_text(errors="ignore")
            scanned += 1
        except Exception:
            pass
print(f"  scanned {scanned} source files across {[k for k, v in pkg_paths.items() if v]}\n")

def _hits(token):
    pat = re.compile(r"\b" + re.escape(token) + r"\b")
    return [f for f, src in corpus.items() if pat.search(src)]

source_evidence = {"files_scanned": scanned, "names": {}, "params": {}, "undeclared": []}
missing_names = []
if corpus:
    print("  TOOL NAMES")
    for n in OUR_NAMES:
        h = _hits(n)
        source_evidence["names"][n] = len(h)
        if h:
            print(f"    ✓ {n:<22} {len(h)} file(s), e.g. {Path(h[0]).name}")
        else:
            print(f"    ✗ {n:<22} NOT FOUND IN HARNESS SOURCE")
            missing_names.append(n)
    print("\n  CRITICAL PARAMETER NAMES")
    for p in CRIT_PARAMS:
        h = _hits(p)
        source_evidence["params"][p] = len(h)
        print(f"    {'✓' if h else '✗'} {p:<22} {len(h)} file(s)")
    cand = set()
    for src in corpus.values():
        cand |= set(re.findall(r'["\']name["\']\s*:\s*["\'](\w+)["\']', src))
        cand |= set(re.findall(r"def\s+(\w*(?:file|patch|command|search|graph|status)\w*)\s*\(", src))
    extra = sorted(c for c in cand if c not in OUR_NAMES and not c.startswith("_"))
    source_evidence["undeclared"] = extra[:60]
    if extra:
        print(f"\n  Harness-side candidate tool identifiers we do NOT declare ({len(extra)}):")
        for c in extra[:40]:
            print(f"      {c}")
else:
    print("  No harness source readable — run Cell 1 first (installs the ADK).")

# ---------------------------------------------------------------- D. the diff
mismatches = []
if real_tools is not None:
    def normalize(t):
        if isinstance(t, dict):
            fn = t.get("function", t)
            params = (fn.get("parameters") or {}).get("properties", {}) or {}
            return fn.get("name", "?"), set(params.keys())
        if callable(t):
            try:
                ps = set(inspect.signature(t).parameters.keys()) - {"self", "cls"}
            except (TypeError, ValueError):
                ps = set()
            return getattr(t, "__name__", str(t)), ps
        decl = getattr(t, "declaration", None) or getattr(t, "_declaration", None)
        if decl is not None:
            params = getattr(decl, "parameters", None)
            props = getattr(params, "properties", {}) if params else {}
            return getattr(decl, "name", "?"), set(props or {})
        return str(t), set()

    real = dict(normalize(t) for t in real_tools)
    ours = dict(normalize(t) for t in COMPETITION_TOOLS)

    print("\n" + "=" * 72)
    print("PHASE 0.2: TOOL SCHEMA DIFF  (real ADK  vs  COMPETITION_TOOLS)")
    print("=" * 72)
    for name in sorted(set(real) | set(ours)):
        if name not in real:
            mismatches.append(f"EXTRA    {name}: declared by us, absent from the harness")
        elif name not in ours:
            mismatches.append(f"MISSING  {name}: harness tool we never declare")
        else:
            only_real, only_ours = real[name] - ours[name], ours[name] - real[name]
            if only_real or only_ours:
                mismatches.append(
                    f"PARAMS   {name}: harness-only={sorted(only_real) or '-'}  "
                    f"ours-only={sorted(only_ours) or '-'}"
                )
            else:
                print(f"  ✓ {name}({', '.join(sorted(real[name]))})")
    if mismatches:
        print("\n  ✗ MISMATCHES FOUND:")
        for m in mismatches:
            print(f"      {m}")
        print("\n  >>> Do NOT train until reconciled. Prefer importing the real schemas")
        print("      over editing the copy — a copy drifts, an import cannot:")
        print("          import chat_format; chat_format.COMPETITION_TOOLS = real_tools")
    else:
        print("\n  ✓ All tool and parameter names match. Safe to train on.")
    print("=" * 72)
    outcome = "mismatch" if mismatches else "clean"
else:
    outcome = "inconclusive"
    print("\n  (No importable tool registry found — rely on the source evidence above.)")
    print("  Candidate list/dict globals, for manual inspection:")
    for _k, _v in sorted(globals().items()):
        if _k.startswith("_") or _k in ("COMPETITION_TOOLS", "corpus", "pkg_paths", "source_evidence"):
            continue
        if isinstance(_v, (list, tuple, dict)) and 3 <= len(_v) <= 40:
            print(f"      {_k}: {type(_v).__name__}[{len(_v)}]")
    print("  If one of those is the tool list: set `real_tools = <it>` and re-run.")

# Any tool name absent from the harness source is a mismatch on its own, even
# when no importable registry exists to diff against.
if missing_names:
    outcome = "mismatch"
    mismatches += [f"SOURCE   {n}: name appears in NO harness source file" for n in missing_names]

# ------------------------------------------ E. F2 probe — ALWAYS runs (v1 bug)
print("\n" + "=" * 72)
print("F2 PROBE: do the declarations survive into rendered training text?")
print("=" * 72)
from transformers import AutoTokenizer
from chat_format import render
_MP = globals().get("MODEL_PATH") or Path(
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"
)
_tok = AutoTokenizer.from_pretrained(str(_MP), local_files_only=True)
f2_ok, f2_tokens = False, None
try:
    _probe = render([{"role": "user", "content": "probe"}], _tok, tools=COMPETITION_TOOLS)
except RuntimeError as e:
    print(f"\n>>> F2 IS LIVE: {e}")
    print(">>> The template drops `tools=`. Training text contains no declarations.")
else:
    f2_tokens = len(_tok(_probe, add_special_tokens=False)["input_ids"])
    f2_ok = all(p in _probe for p in ("old_string", "node_id", "submit_patch"))
    print(f"Rendered preamble with tools: {f2_tokens} tokens")
    print(f"Declarations present in text: {f2_ok}")
    if f2_ok:
        print(">>> F2 is FIXED. Record this token count — it is manuscript §5.2 evidence.")
    else:
        print(">>> F2 IS LIVE: template ignored `tools=` without raising.")
print("=" * 72)

# ------------------------------------------------------- F. write the receipt
_p = write_tool_schema_receipt(
    outcome,
    f2_ok=f2_ok,
    f2_preamble_tokens=f2_tokens,
    discovery=found_via or "none",
    mismatches=mismatches,
    source_evidence=source_evidence,
)
print(f"\nReceipt: {_p}  outcome={outcome}  f2_ok={f2_ok}")
if outcome == "clean" and f2_ok:
    print(">>> Gate will PASS. Training may proceed.")
elif outcome == "mismatch" or not f2_ok:
    print(">>> Gate will REFUSE training until this is reconciled.")
else:
    print(">>> Gate will REFUSE training unless you set os.environ['TOOL_SCHEMA_ACK']='1',")
    print("    which records a deliberate decision to train on unconfirmed schemas.")
```

---

#### Cell 0D: `[Phase 0.2 drill-down]` Resolve the `node_id` Finding

> **Run only if Cell 0C flagged something.** Cell 0C reported all 9 tool names present
> in the harness source but `node_id` in **zero** `.py` files. `node_id` is the
> *required* parameter of `get_code_subgraph` and `get_code_neighbors` — the two tools
> `DEFAULT_SYSTEM_PROMPT` orders the model to call first. Two caveats before concluding:
> Cell 0C scanned only `*.py`, and a grep is evidence of absence only where it looked.
> This cell widens the net to JSON/YAML/Jinja and then reads the real signatures out of
> `agent_runner.py`, which matched 6 of our 9 tools.

```python
# =============================================================================
# Cell 0D — Drill down on the `node_id` finding.
#
# Cell 0C reported all 9 tool names present in the harness source, but
# `node_id` in ZERO .py files. `node_id` is the REQUIRED parameter of
# get_code_subgraph and get_code_neighbors — the two tools the system prompt
# orders the model to call FIRST. If the harness calls that parameter
# something else, every graph call fails argument validation, the budget
# burns on retries, and what reaches submit_patch is an unmodified tree.
#
# Two things to settle:
#   1. Cell 0C only scanned *.py. If the schemas live in JSON/YAML, `node_id`
#      could be present and simply not greppable there. Widen the net.
#   2. agent_runner.py matched 6 of our 9 tools. That is the file that defines
#      the real tool surface. Read the actual signatures out of it.
# =============================================================================
import re, sys, json, inspect, importlib
from pathlib import Path
sys.path.insert(0, "/kaggle/working")
from chat_format import COMPETITION_TOOLS

PKGS = ("swegemma", "adk_eval_core", "adk_submission")
roots = {}
for _p in PKGS:
    try:
        roots[_p] = Path(importlib.import_module(_p).__file__).parent
    except Exception:
        pass

# ---------------------------------------------------------- 1. widen the net
print("=" * 72)
print("1. NON-PYTHON SCHEMA FILES — is `node_id` declared outside .py?")
print("=" * 72)
EXTS = ("*.json", "*.yaml", "*.yml", "*.toml", "*.txt", "*.md", "*.jinja", "*.j2")
other = {}
for _r in roots.values():
    for pat in EXTS:
        for f in _r.rglob(pat):
            try:
                other[str(f)] = f.read_text(errors="ignore")
            except Exception:
                pass
print(f"  scanned {len(other)} non-.py files")
for tok in ("node_id", "node", "symbol", "symbol_id", "qualified_name", "entity_id", "name"):
    hit = [f for f, s in other.items() if re.search(r"\b" + tok + r"\b", s)]
    print(f"    {'✓' if hit else '✗'} {tok:<16} {len(hit)} file(s)"
          + (f"  e.g. {Path(hit[0]).name}" if hit else ""))

# ------------------------------------------- 2. read the real tool surface
print("\n" + "=" * 72)
print("2. REAL TOOL SIGNATURES FROM THE HARNESS")
print("=" * 72)
OUR = {}
for t in COMPETITION_TOOLS:
    f = t.get("function", t)
    props = (f.get("parameters") or {}).get("properties", {}) or {}
    OUR[f["name"]] = (set(props), set((f.get("parameters") or {}).get("required", [])))

targets = [f for r in roots.values() for f in r.rglob("*.py")
           if f.name in ("agent_runner.py", "tools.py", "agent.py", "toolset.py")]
# plus any file that mentions >=3 of our tool names
for r in roots.values():
    for f in r.rglob("*.py"):
        if f in targets:
            continue
        try:
            s = f.read_text(errors="ignore")
        except Exception:
            continue
        if sum(1 for n in OUR if re.search(r"\b" + n + r"\b", s)) >= 3:
            targets.append(f)
print(f"  candidate definition files: {[t.name for t in targets]}\n")

DEF_RE = re.compile(
    r"^\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", re.M)
for f in targets:
    try:
        src = f.read_text(errors="ignore")
    except Exception:
        continue
    found = []
    for m in DEF_RE.finditer(src):
        fname, args = m.group(1), m.group(2)
        base = fname[:-5] if fname.endswith("_tool") else fname
        if base in OUR or fname in OUR:
            params = [a.split(":")[0].split("=")[0].strip()
                      for a in args.split(",") if a.strip()]
            params = [p for p in params if p not in ("self", "cls", "*", "**kwargs", "ctx")]
            found.append((base, fname, params))
    if found:
        print(f"  --- {f} ---")
        for base, fname, params in found:
            ours_props, ours_req = OUR.get(base, (set(), set()))
            extra = [p for p in params if p not in ours_props]
            missing = [p for p in sorted(ours_props) if p not in params]
            flag = "✓" if not extra and not missing else "✗"
            print(f"    {flag} {fname}({', '.join(params)})")
            if extra:
                print(f"        harness-only params : {extra}")
            if missing:
                print(f"        we declare, absent  : {missing}")
        print()

# ------------------- 3. verbatim context around the graph tools (the suspects)
print("=" * 72)
print("3. VERBATIM CONTEXT: get_code_subgraph / get_code_neighbors")
print("=" * 72)
shown = 0
for f in targets:
    try:
        lines = f.read_text(errors="ignore").split("\n")
    except Exception:
        continue
    for i, ln in enumerate(lines):
        if re.search(r"\bget_code_(subgraph|neighbors)\b", ln):
            lo, hi = max(0, i - 3), min(len(lines), i + 12)
            print(f"\n  {f.name}:{i+1}")
            for j in range(lo, hi):
                print(f"    {j+1:>5} | {lines[j]}")
            shown += 1
            break
    if shown >= 4:
        break
if not shown:
    print("  (no match in candidate files — widen `targets` above)")
print("\n" + "=" * 72)
print("READ THE PARAMETER NAME ABOVE. If it is not `node_id`, that is the")
print("score-0 cause: both graph tools reject every call the policy makes.")
print("=" * 72)
```

---

#### Cell 0E: `[Phase 0.2 drill-down]` Read the Authoritative Tool Surface

> **Cell 0D §1 was a null result, and that is informative.** It scanned exactly **one**
> non-`.py` file across all three packages. Finding nothing there is not evidence that
> `node_id` is absent — the scan had nowhere to look. What it does establish is that the
> harness ships no JSON/YAML schema files, so the Python source *is* the entire tool
> surface, and Cell 0C's `node_id` = 0 across 98 `.py` files stands unqualified.

> Cell 0D's candidate list also named **`registry.py`**. Cell 0C recorded
> `outcome=inconclusive` because it found no module-level tool list — but a file called
> `registry.py` means a registry exists in some other shape. Recovering it flips the
> receipt to `clean`/`mismatch` and removes the need for `TOOL_SCHEMA_ACK`.

```python
# =============================================================================
# Cell 0E — Read the authoritative tool surface. Two targets:
#
#   graph.py     the two tools whose required parameter `node_id` appears
#                nowhere in the harness. Print their real signatures and
#                docstrings verbatim.
#   registry.py  Cell 0C found no importable registry and recorded
#                outcome=inconclusive, which makes the gate refuse training.
#                A file literally named registry.py says one exists — it was
#                simply not a module-level list. Find it and the outcome can
#                become clean/mismatch instead of inconclusive.
# =============================================================================
import re, sys, inspect, importlib, pkgutil, json
from pathlib import Path
sys.path.insert(0, "/kaggle/working")
from chat_format import COMPETITION_TOOLS

roots = {}
for _p in ("swegemma", "adk_eval_core", "adk_submission"):
    try:
        roots[_p] = Path(importlib.import_module(_p).__file__).parent
    except Exception:
        pass

def _find(fname):
    return [f for r in roots.values() for f in r.rglob(fname)]

# --------------------------------------------- 1. the graph tools, verbatim
print("=" * 72)
print("1. GRAPH TOOLS — verbatim source")
print("=" * 72)
GRAPH = ("get_code_subgraph", "get_code_neighbors")
seen = set()
for f in [x for n in ("graph.py", "tools.py", "agent_runner.py") for x in _find(n)]:
    if f in seen:
        continue
    seen.add(f)
    try:
        lines = f.read_text(errors="ignore").split("\n")
    except Exception:
        continue
    for i, ln in enumerate(lines):
        m = re.match(r"\s*(?:async\s+)?def\s+(\w+)", ln)
        if not m:
            continue
        if not any(g in m.group(1) for g in GRAPH):
            continue
        # print until the def's body is clearly past the signature + docstring
        print(f"\n--- {f}:{i+1} ---")
        depth, j, printed = 0, i, 0
        while j < len(lines) and printed < 40:
            print(f"  {j+1:>5} | {lines[j]}")
            depth += lines[j].count("(") - lines[j].count(")")
            printed += 1
            j += 1
            if printed > 1 and depth <= 0 and re.match(r"\s*(return|[a-z_]+\s*=|if |for |try)", lines[j] if j < len(lines) else ""):
                break

# ------------------------------------------------ 2. the registry, resolved
print("\n" + "=" * 72)
print("2. TOOL REGISTRY")
print("=" * 72)
for f in _find("registry.py"):
    print(f"\n--- {f} ---")
    try:
        src = f.read_text(errors="ignore")
    except Exception:
        continue
    for i, ln in enumerate(src.split("\n")[:120]):
        print(f"  {i+1:>4} | {ln}")

# Try to pull the live registry object out of the imported module.
print("\n  Live registry objects:")
for _p, _root in roots.items():
    for _mi in pkgutil.walk_packages([str(_root)], prefix=_p + "."):
        if not _mi.name.endswith((".registry", ".tools", ".graph")):
            continue
        try:
            m = importlib.import_module(_mi.name)
        except Exception as e:
            print(f"    {_mi.name}: import failed ({e!r})")
            continue
        for k, v in vars(m).items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and v and all(isinstance(x, str) for x in v):
                print(f"    {_mi.name}.{k}: dict[{len(v)}] keys={sorted(v)[:12]}")
            elif isinstance(v, (list, tuple)) and 3 <= len(v) <= 40:
                print(f"    {_mi.name}.{k}: {type(v).__name__}[{len(v)}]")
            elif callable(v) and (k.endswith("_tool") or k in
                                  {t.get('function', t)['name'] for t in COMPETITION_TOOLS}):
                try:
                    print(f"    {_mi.name}.{k}{inspect.signature(v)}")
                except (TypeError, ValueError):
                    print(f"    {_mi.name}.{k}(<signature unavailable>)")

print("\n" + "=" * 72)
print("WHAT TO DO WITH THIS")
print("=" * 72)
print("  - If the graph tools take a parameter other than `node_id`, that is the")
print("    score-0 cause: the policy's first move fails validation every time.")
print("    Fix COMPETITION_TOOLS to match, re-run Cell 0C, then train.")
print("  - If a live registry turned up above, set `real_tools = <it>` at the top")
print("    of Cell 0C and re-run: the outcome becomes clean/mismatch rather than")
print("    inconclusive, and the gate stops needing TOOL_SCHEMA_ACK.")
print("=" * 72)
```

---

#### Cell 0F: `[Phase 0.2 RESOLUTION]` Generate `COMPETITION_TOOLS` From the Live Harness

> **The mismatch is confirmed.** `swegemma/tools/graph.py` documents
> `get_code_neighbors(ctx, node, edge_type=None, ...)` — *"node: Dotted symbol name or
> suffix of the function, method, or class node"*. We declare `node_id`, and a
> `direction` parameter with an `in`/`out`/`both` enum the harness has no concept of.
> Both parameters of a graph tool are wrong, and the system prompt orders the model to
> call a graph tool **first**.

> **Do not hand-edit the schemas.** Hand-transcription is the defect. `swegemma.tools`
> re-exports all nine tools as callables, so the schemas can be *derived* from
> `inspect.signature` plus the docstrings. That is what this cell does.

```python
# =============================================================================
# Cell 0F — Generate COMPETITION_TOOLS from the live harness, mechanically.
#
# Cell 0E found the registry: swegemma/tools/__init__.py re-exports all nine
# tools as plain callables. Cell 0C never found it because it is a module
# namespace, not a module-level list.
#
# The get_code_neighbors docstring settles the open question:
#     node: Dotted symbol name or suffix ('pkg.module.Class.method', ...)
# We declare `node_id`. We also declare `direction` with an in/out/both enum;
# the harness takes `edge_type`. Both parameters of a graph tool are wrong.
#
# Do NOT hand-edit COMPETITION_TOOLS to fix this. Hand-transcription is what
# produced the defect. This cell derives the schemas from inspect.signature
# and the docstrings, prints them, and writes them to disk.
# =============================================================================
CELL_0F_VERSION = "v2 (2026-11-10) — container-aware JSON typing, versioned output"
print(f"CELL 0F {CELL_0F_VERSION}\n")

import inspect, json, re, sys
from pathlib import Path
sys.path.insert(0, "/kaggle/working")
from chat_format import COMPETITION_TOOLS

import swegemma.tools as T

OUR = {}
for t in COMPETITION_TOOLS:
    f = t.get("function", t)
    params = (f.get("parameters") or {})
    OUR[f["name"]] = (set((params.get("properties") or {})), set(params.get("required") or []))

# Parameters the harness injects and the model never supplies.
INJECTED = {"ctx", "self", "cls", "context", "workspace", "sandbox", "graph", "state"}

# NB: substring matching against str(annotation) is a trap — "list[str]" contains
# "str", so a naive scan types a list-of-strings as "string". Containers are therefore
# tested FIRST, and scalars with word boundaries. Getting this wrong reproduces F6:
# the model would emit a bare string where the harness expects a list.
SCALARS = (("bool", "boolean"), ("int", "integer"), ("float", "number"), ("str", "string"))

def _docstring_args(fn):
    """Pull the Google-style `Args:` block into {param: description}."""
    doc = inspect.getdoc(fn) or ""
    out, cur = {}, None
    in_args = False
    for ln in doc.split("\n"):
        if re.match(r"\s*Args:\s*$", ln):
            in_args = True
            continue
        if in_args and re.match(r"\s*(Returns|Raises|Yields|Examples?):\s*$", ln):
            break
        if not in_args:
            continue
        m = re.match(r"\s{0,8}(\w+)\s*(?:\([^)]*\))?:\s*(.*)$", ln)
        if m:
            cur = m.group(1)
            out[cur] = m.group(2).strip()
        elif cur and ln.strip():
            out[cur] += " " + ln.strip()
    return out

def _json_type(ann):
    """Return a JSON-schema fragment, e.g. {"type":"array","items":{"type":"string"}}."""
    if ann is inspect.Parameter.empty:
        return {"type": "string"}
    txt = str(ann).replace("typing.", "")
    txt = re.sub(r"\|\s*None", "", txt)
    txt = re.sub(r"Optional\[(.*)\]", r"\1", txt).strip()
    low = txt.lower()
    if re.match(r"(list|sequence|tuple|set|iterable)\b", low):
        inner = "string"
        m = re.search(r"\[([^\],]+)", txt)
        if m:
            it = m.group(1).strip().lower()
            inner = dict(SCALARS).get(it, "string")
        return {"type": "array", "items": {"type": inner}}
    if re.match(r"(dict|mapping)\b", low):
        return {"type": "object"}
    for key, js in SCALARS:
        if re.search(r"\b" + key + r"\b", low):
            return {"type": js}
    return {"type": "string"}

generated, report = [], []
for name in OUR:
    fn = getattr(T, name, None)
    if fn is None:
        report.append(f"  ✗ {name}: NOT exported by swegemma.tools")
        continue
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        report.append(f"  ✗ {name}: signature unavailable ({e!r})")
        continue
    docs = _docstring_args(fn)
    props, required = {}, []
    for pname, p in sig.parameters.items():
        if pname in INJECTED or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        props[pname] = {**_json_type(p.annotation),
                        "description": docs.get(pname, "")}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    generated.append({"type": "function", "function": {
        "name": name,
        "description": (inspect.getdoc(fn) or "").split("\n")[0],
        "parameters": {"type": "object", "properties": props, "required": required},
    }})
    ours_props, ours_req = OUR[name]
    only_harness = sorted(set(props) - ours_props)
    only_ours = sorted(ours_props - set(props))
    mark = "✓" if not only_harness and not only_ours else "✗"
    report.append(f"  {mark} {name}{sig}")
    if only_harness:
        report.append(f"      harness-only : {only_harness}")
    if only_ours:
        report.append(f"      ours-only    : {only_ours}  <-- the model is trained to send these")

print("=" * 72)
print("LIVE SIGNATURES vs COMPETITION_TOOLS")
print("=" * 72)
print("\n".join(report))

GENERATOR_VERSION = "2026-11-10.0F-v2"   # must match chat_format.REQUIRED_REAL_TOOLS_GENERATOR
out = Path("/kaggle/working/real_tools.json")
out.write_text(json.dumps({"generator_version": GENERATOR_VERSION, "tools": generated}, indent=2))
print(f"\n✓ Generated {len(generated)} schemas -> {out} (generator {GENERATOR_VERSION})")

# Make Cell 0C's diff authoritative instead of inconclusive.
real_tools = [getattr(T, n) for n in OUR if getattr(T, n, None) is not None]
print(f"✓ `real_tools` set ({len(real_tools)} callables). Re-run Cell 0C now:")
print("    the receipt becomes clean/mismatch instead of inconclusive.")

print("\n" + "=" * 72)
print("GENERATED SCHEMAS (paste over COMPETITION_TOOLS in src/chat_format.py)")
print("=" * 72)
print(json.dumps(generated, indent=4))

print("\n" + "=" * 72)
print("TWO THINGS THAT MUST CHANGE WITH THE SCHEMAS")
print("=" * 72)
print("  1. TOOL_RENDER_PROBES still contains 'node_id'. Once the schemas use the")
print("     real parameter name, that probe no longer appears in rendered text and")
print("     render() will raise a FALSE F2 failure. Update the probes in the same")
print("     edit as the schemas.")
print("  2. The training data carries the wrong name too. Both generators emit")
print("     `\"arguments\": {\"node_id\": ...}` (build_real_sft.py:219,")
print("     teacher_rollouts.py:271), so every trajectory teaches the rejected")
print("     call. Rewriting the schemas alone does not fix the dataset — see the")
print("     rewrite pass in TRAINING_FIX_PLAN_DEC2.md Stage 1.5. NOTE: for")
print("     get_code_subgraph this is NOT a rename. The harness takes")
print("     `nodes: list[str]`, so {\"node_id\": x} becomes {\"nodes\": [x]} —")
print("     the VALUE changes shape, not just the key.")

# chat_format caches COMPETITION_TOOLS at import time, so the file just written is not
# in effect until the module is reloaded. Do it here rather than relying on the operator.
import importlib, chat_format
importlib.reload(chat_format)
print("\n" + "=" * 72)
print(f"chat_format reloaded. Schemas now from: {chat_format.TOOLS_SOURCE}")
print(f"Fingerprint: {chat_format.tool_schema_fingerprint()}")
print("Re-run Cell 0C now to write a receipt against THESE schemas.")
print("=" * 72)
print("=" * 72)
```

---


---

### PART 2: PHASE 4.1 SFT POST-TRAINING (NO RFT OVERWRITE)

#### Cell 5: `[RepoGraphForge]` GPU VRAM Flush & Health Check
```python
import os, torch

print("=== FLUSHING GPU MEMORY ===")
!fuser -k 8000/tcp 2>/dev/null
!pkill -9 -f vllm 2>/dev/null
!pkill -9 -f "Worker_TP" 2>/dev/null
!pkill -9 -f multiprocessing 2>/dev/null
torch.cuda.empty_cache()

num_gpus = torch.cuda.device_count()
print(f"✓ Detected {num_gpus} GPUs:")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

!nvidia-smi
```

---

#### Cell 6: `[Phase 1 & Phase 4.1]` Write `train_sft.py` Using `chat_format.py` (`MAX_LEN=8192`, `lr=1e-4`, `2 Epochs`)
*(Note: `%%writefile` must be on Line 1 of this cell)*

```python
%%writefile /kaggle/working/train_sft.py
import os, sys, json, logging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, "/kaggle/working")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_sft")

import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False

from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from chat_format import build_example, verify_and_print_samples, COMPETITION_TOOLS, DEFAULT_SYSTEM_PROMPT

# --- Phase 0.2 HARD GATE -----------------------------------------------------
# Refuses to start unless Cell 0C has verified COMPETITION_TOOLS against the real
# harness, against THIS chat_format version and THESE exact schemas. Training on
# wrong tool schemas costs ~3 GPU hours and yields a run that converges, completes,
# submits patches and scores zero — with no error at training time. This is the
# last point at which that can be caught.
from chat_format import assert_tool_schema_verified
_receipt = assert_tool_schema_verified()
logger.info(
    "Tool schemas verified: outcome=%s, preamble=%s tokens, discovery=%s",
    _receipt["outcome"], _receipt.get("f2_preamble_tokens"), _receipt.get("discovery"),
)
# -----------------------------------------------------------------------------

MODEL_PATH = Path("/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2")
TRAJECTORIES_FILE = Path("/kaggle/working/data/trajectories/sft_combined.jsonl")
if not TRAJECTORIES_FILE.exists():
    for fb in ("sft_real_train.jsonl", "sft_verified.jsonl"):
        p = Path("/kaggle/working/data/trajectories") / fb
        if p.exists():
            TRAJECTORIES_FILE = p
            break

OUTPUT_ADAPTER_DIR = Path("/kaggle/working/submission/adapters/main_lora")
OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Cap MAX_LEN at 2560 so the 256k-vocab logits tensor never exceeds 2.5 GiB!
MAX_LEN = 2560
raw_records = []
with open(TRAJECTORIES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            raw_records.append(json.loads(line))

kept = []
dropped_overflow = 0
for rec in raw_records:
    msgs = rec.get("messages", [])
    if not msgs:
        continue
    ex = build_example(msgs, tokenizer, max_len=MAX_LEN, tools=COMPETITION_TOOLS, system_prompt=DEFAULT_SYSTEM_PROMPT)
    if ex is None:
        dropped_overflow += 1
    else:
        kept.append(ex)

logger.info("✓ Tokenized %d trajectories (dropped %d > %d tokens).", len(kept), dropped_overflow, MAX_LEN)

# --- Dataset survival gate -----------------------------------------------------------
# Drop-don't-truncate is only safe at a generous budget. Combined with a large fixed
# preamble and a reduced MAX_LEN it silently becomes a dataset filter. Fail loudly.
DATASET_FLOOR = 60
drop_rate = dropped_overflow / max(1, len(raw_records))
assert len(kept) >= DATASET_FLOOR, (
    f"Only {len(kept)} trajectories survived preprocessing (floor={DATASET_FLOOR}, "
    f"dropped {dropped_overflow}). Raise MAX_LEN or left-truncate — do not train on this."
)
if drop_rate > 0.10:
    logger.warning(
        "⚠️  %.1f%% of trajectories dropped at MAX_LEN=%d. The surviving set is biased "
        "toward short trajectories; raise MAX_LEN.", drop_rate * 100, MAX_LEN,
    )

verify_and_print_samples(kept, tokenizer, num_samples_to_print=3)
tokenized_ds = Dataset.from_list(kept)

# --- Optimizer-step gate -------------------------------------------------------------
# A rank-32 LoRA trained for 2-4 optimizer steps is noise, not training.
_steps = (len(kept) * 2) // 8   # num_train_epochs=2, grad_accum=8
assert _steps >= 30, (
    f"Schedule yields only {_steps} optimizer steps. Lower gradient_accumulation_steps "
    f"or add data before spending GPU quota."
)
logger.info("✓ Schedule yields %d optimizer steps across 2 epochs.", _steps)

# Asymmetric memory allocation: Gives GPU 0 massive headroom (~10 GiB free) while ensuring
# total capacity (75 GiB) easily fits the 62.5 GB model with zero layers offloaded to meta!
num_gpus = torch.cuda.device_count()
if num_gpus == 4:
    max_memory = {0: "12.0GiB", 1: "21.0GiB", 2: "21.0GiB", 3: "21.0GiB"}
elif num_gpus > 1:
    max_memory = {0: "12.0GiB", 1: "21.0GiB"}
else:
    max_memory = None

base_model = Gemma4ForConditionalGeneration.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory=max_memory,
    local_files_only=True,
)

TEXT_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
text_linear_targets = [
    name for name, mod in base_model.named_modules()
    if name.endswith(TEXT_SUFFIXES)
    and isinstance(mod, nn.Linear)
    and "vision" not in name.lower()
    and "audio" not in name.lower()
]

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=text_linear_targets,
)

base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()
model = get_peft_model(base_model, peft_config)
logger.info("✓ LoRA Adapter Attached across %d text linear targets:", len(text_linear_targets))
model.print_trainable_parameters()

# Phase 4.1: lr=1e-4, num_train_epochs=2 (prevents overfitting/overwriting base capabilities)
training_args = TrainingArguments(
    output_dir="/kaggle/working/sft_checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=2,
    logging_steps=5,
    save_strategy="no",
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_ds,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
)

logger.info("🚀 STARTING PHASE 4.1 SFT TRAINING (2 Epochs, lr=1e-4, max_len=%d)...", MAX_LEN)
trainer.train()

trainer.model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
tokenizer.save_pretrained(str(OUTPUT_ADAPTER_DIR))

for f in OUTPUT_ADAPTER_DIR.glob("*.jinja"): f.unlink()
for f in OUTPUT_ADAPTER_DIR.glob("README.md"): f.unlink()

logger.info("✓ SFT TRAINING COMPLETE! Adapter exported to %s", OUTPUT_ADAPTER_DIR)
```

---

#### Cell 7: `[Phase 4.1 Execution]` Launch SFT Training (Inspect the Phase 1.2 Printed Output at Top!)
```python
!python3 /kaggle/working/train_sft.py
```

---

### PART 3: PHASE 4.2 BENCHMARK, 3-ARM EVALUATION & AUDIT

#### Cell 8: `[Phase 3.3, 3.4 & 4.2]` Sync Adapter, Set `max_output_tokens: 4096` & Select Evaluation Arm (`A`, `B`, or `C`)
*(Supports all 3 evaluation arms from `FIX_PLAN.md` Phase 4.2: `'A'` = baseline RFT v1 snapshot + old prompt, `'B'` = new SFT adapter + old anti-hesitation prompt, `'C'` = new SFT adapter + new Phase 3.4 budget prompt)*

```python
import shutil
from pathlib import Path
from adk_submission import discover_adapters, discover_declared_models
from chat_format import DEFAULT_SYSTEM_PROMPT

WORKING_DIR = Path("/kaggle/working")
AGENT_DIR = WORKING_DIR / "sample_submission"
dst_adapter = AGENT_DIR / "adapters/main_lora"

# Select Arm from FIX_PLAN.md Phase 4.2:
#   "C" (Recommended): New SFT adapter + Phase 3.4 Budget Prompt (reads <=3 files, runs pytest)
#   "B": New SFT adapter + legacy anti-hesitation prompt
#   "A": Baseline rft_v1_baseline snapshot + legacy anti-hesitation prompt
EVAL_ARM = "C"

if EVAL_ARM == "A":
    src_adapter = WORKING_DIR / "experiments/adapters/rft_v1_baseline"
else:
    src_adapter = WORKING_DIR / "submission/adapters/main_lora"

# 1. Sync selected adapter into sample_submission
dst_adapter.mkdir(parents=True, exist_ok=True)
for f in src_adapter.glob("*"):
    if f.is_file() and not f.name.endswith(".jinja") and not f.name.endswith(".md"):
        shutil.copy2(f, dst_adapter / f.name)

# 2. Sanitize
for p in [src_adapter, dst_adapter]:
    if p.exists():
        for f in p.glob("*.jinja"): f.unlink()
        for f in p.glob("README.md"): f.unlink()

# 3. Configure System Prompt according to EVAL_ARM
prompt_file = AGENT_DIR / "prompts/system.md"
prompt_file.parent.mkdir(parents=True, exist_ok=True)

LEGACY_ANTI_HESITATION_PROMPT = """You are an autonomous software engineering agent equipped with a pre-computed repository code graph.

MANDATORY EXECUTION CONSTRAINTS:
1. MAX 1 FILE READ: Once you read the primary file where the bug occurs (e.g. models.py), you are STRICTLY FORBIDDEN from reading secondary helper files, compatibility shims (like compat.py), or test files.
2. DO NOT SECOND-GUESS: Do not over-analyze unrelated symbols or utility functions. Focus ONLY on the exact symptom reported in the problem statement.
3. MANDATORY EDIT ON TURN 3: You MUST invoke `edit_file` immediately after reading the target function. Never emit thoughts without a tool call.
4. IMMEDIATE SUBMISSION: As soon as `edit_file` succeeds, your next turn MUST be `submit_patch`.
""".strip()

if EVAL_ARM == "C":
    prompt_file.write_text(DEFAULT_SYSTEM_PROMPT.strip())
else:
    prompt_file.write_text(LEGACY_ANTI_HESITATION_PROMPT)

# 4. Fix C3: Raise max_output_tokens from 1024 -> 4096 so multi-hunk edits/patches never truncate!
sampling_file = AGENT_DIR / "configs/sampling.yaml"
sampling_file.parent.mkdir(parents=True, exist_ok=True)
sampling_file.write_text("temperature: 0.1\ntop_p: 0.95\nmax_output_tokens: 4096\n")

# 5. Configure agent.yaml
agent_yaml = AGENT_DIR / "agent.yaml"
agent_yaml.write_text("""name: swe_baseline_agent
model: gemma-4-31b-it-qat-w4a16-ct
adapter: main_lora
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml
      skip_summarization: true
generate_content_config: !include configs/sampling.yaml
""".strip())

print(f"=== SYNC COMPLETE (EVAL_ARM = {EVAL_ARM}) ===")
print(f"✓ Adapter Source:     {src_adapter}")
print(f"✓ Adapter Size:       {(dst_adapter / 'adapter_model.safetensors').stat().st_size / 1e6:.2f} MB")
print(f"✓ max_output_tokens:  4096 (Fixes C3)")
print(f"✓ Declared Model:     {discover_declared_models(AGENT_DIR)}")
print(f"✓ Declared Adapter:   {discover_adapters(AGENT_DIR)}")
print("=" * 50)
```

---

#### Cell 9: `[Section 4 from Ryan's Notebook]`
*Click Run on Section 4 to start vLLM serving `main_lora` on port 8000.*

---

#### Cell 10: `[Section 5 from Ryan's Notebook]` Configure & Run 14-Task Holdout Evaluation Loop
```python
# In Section 5:
TASKS_PATH = Path('/kaggle/working/data/splits/val_ood_tasks.jsonl')
tasks = load_tasks(TASKS_PATH)
SAMPLE_TASKS = tasks

timeout_seconds = 300
verification_timeout_seconds = 300
max_turns = 12
```
*Click Run on the evaluation loop cell directly below it!*

---

#### Cell 11: `[Phase 3.5 & 4.2]` Corrected Holdout Benchmark Audit Table (Fixes C5 + Logs `hit_max_output_tokens`)
*(Deletes the buggy `or args == {}` check so zero-arg tools like `get_status()` are never miscounted as `submit_patch`, and tracks how many responses hit `max_output_tokens` to verify B1)*

```python
import json, subprocess, tempfile
from pathlib import Path
import pandas as pd

TRACES_DIR = Path("/kaggle/working/results/traces")
pred_map = {p["id"]: p.get("prediction", "") for p in predictions} if "predictions" in globals() else {}

# --- Discover the harness's resolution verdicts --------------------------------------
# Section 5 runs verification (`verification_timeout_seconds`), so a pass/fail verdict
# exists somewhere in its output. Field names vary by harness version, so probe for them
# rather than guessing, and PRINT what was found so the mapping can be corrected once.
RESOLVED_KEYS = ("resolved", "passed", "is_resolved", "success", "test_passed", "exit_code")

def _extract_resolved(obj):
    if not isinstance(obj, dict):
        return None
    for k in RESOLVED_KEYS:
        if k in obj:
            v = obj[k]
            if k == "exit_code":
                return v == 0
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("true", "pass", "passed", "resolved", "1")
            if isinstance(v, (int, float)):
                return bool(v)
    for nested in ("verification", "evaluation", "result", "final_metrics", "report"):
        if isinstance(obj.get(nested), dict):
            r = _extract_resolved(obj[nested])
            if r is not None:
                return r
    return None

resolved_map = {}
for _src_name in ("results", "eval_results", "predictions", "report"):
    _src = globals().get(_src_name)
    if isinstance(_src, list):
        for _row in _src:
            if isinstance(_row, dict):
                _tid = _row.get("id") or _row.get("instance_id") or _row.get("task_id")
                if _tid and _tid not in resolved_map:
                    _r = _extract_resolved(_row)
                    if _r is not None:
                        resolved_map[_tid] = _r
    elif isinstance(_src, dict):
        for _tid, _row in _src.items():
            if _tid not in resolved_map:
                _r = _extract_resolved(_row)
                if _r is not None:
                    resolved_map[_tid] = _r

if not resolved_map:
    print("⚠️  No resolution verdicts found. Inspect the eval output and map the field:")
    for _name in ("results", "eval_results", "predictions", "report"):
        _s = globals().get(_name)
        if isinstance(_s, list) and _s and isinstance(_s[0], dict):
            print(f"    {_name}[0] keys -> {sorted(_s[0].keys())}")
        elif isinstance(_s, dict) and _s:
            _k0 = next(iter(_s))
            print(f"    {_name}[{_k0!r}] keys -> "
                  f"{sorted(_s[_k0].keys()) if isinstance(_s[_k0], dict) else type(_s[_k0])}")
    print("    Then add the right name to RESOLVED_KEYS above.")
else:
    print(f"✓ Resolution verdicts found for {len(resolved_map)} task(s).")

# Repo checkouts for `git apply --check`. Point this at wherever Section 5 unpacks them;
# leave empty to report applies-cleanly as unknown rather than guessing.
REPO_SNAPSHOTS = globals().get("REPO_SNAPSHOTS", {})

records = []
for p in sorted(TRACES_DIR.glob("trace_*.json")):
    data = json.loads(p.read_text())
    t_id = p.stem.replace("trace_", "")
    steps = data.get("steps", [])
    metrics = data.get("final_metrics", {})

    tool_call_count = 0
    files_edited = set()
    called_submit = False
    max_tok_hits = 0

    for s in steps:
        step_metrics = s.get("metrics") or {}
        comp_toks = step_metrics.get("completion_tokens", 0)
        if comp_toks >= 1020:  # Tracks if generation ran into old 1024 or new 4096 ceiling (B1 readout)
            max_tok_hits += 1

        for tc in s.get("tool_calls", []):
            tool_call_count += 1
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name") or tc.get("tool_name") or tc.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception: pass
            if isinstance(args, dict) and "old_string" in args and "filepath" in args:
                files_edited.add(args["filepath"])
            # C5 FIX: Check strictly for submit_patch tool name (NEVER `or args == {}`!)
            if name == "submit_patch":
                called_submit = True

    patch_str = pred_map.get(t_id, "")
    has_nonempty_patch = bool(patch_str and "NO_PATCH" not in patch_str and len(patch_str.strip()) > 0)

    # --- Does the diff even apply? (independent of the harness) ---
    applies = None
    if has_nonempty_patch:
        repo_dir = REPO_SNAPSHOTS.get(t_id)
        if repo_dir and Path(repo_dir).exists():
            with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as fh:
                fh.write(patch_str if patch_str.endswith("\n") else patch_str + "\n")
                diff_path = fh.name
            rc = subprocess.run(
                ["git", "apply", "--check", diff_path],
                cwd=repo_dir, capture_output=True, text=True,
            )
            applies = (rc.returncode == 0)
            Path(diff_path).unlink(missing_ok=True)

    # --- Did it actually resolve the task? (from the harness's own verification) ---
    resolved = resolved_map.get(t_id)

    records.append({
        "Task ID": t_id,
        "Tool Calls": tool_call_count,
        "Called submit_patch": "✓" if called_submit else "✗",
        "Non-Empty Diff": "✓" if has_nonempty_patch else "✗",
        "Applies Cleanly": {True: "✓", False: "✗", None: "?"}[applies],
        "RESOLVED": {True: "✓ PASS", False: "✗ FAIL", None: "? unknown"}[resolved],
        "Patch Chars": len(patch_str) if has_nonempty_patch else 0,
        "Hit Max Tokens": max_tok_hits,
        "Files Edited": list(files_edited) if files_edited else "None",
        "Total Tokens": metrics.get("total_tokens", 0),
    })

df = pd.DataFrame(records)
display(df)

n = max(1, len(df))
n_sub     = int((df["Called submit_patch"] == "✓").sum())
n_valid   = int((df["Non-Empty Diff"] == "✓").sum())
n_applies = int((df["Applies Cleanly"] == "✓").sum())
n_resolved= int((df["RESOLVED"] == "✓ PASS").sum())
n_unknown = int((df["RESOLVED"] == "? unknown").sum())

print("=" * 62)
print(f"  submit_patch called : {n_sub:>3} / {len(df)}  ({n_sub/n*100:5.1f}%)   [attempt]")
print(f"  non-empty diff      : {n_valid:>3} / {len(df)}  ({n_valid/n*100:5.1f}%)   [attempt]")
print(f"  applies cleanly     : {n_applies:>3} / {len(df)}  ({n_applies/n*100:5.1f}%)   [quality]")
print(f"  >>> RESOLVED        : {n_resolved:>3} / {len(df)}  ({n_resolved/n*100:5.1f}%)   [THE METRIC]")
print("=" * 62)
print(f"  steps hitting token ceiling (F1 readout): {int(df['Hit Max Tokens'].sum())}")
if n_unknown:
    print(f"\n  ⚠️  {n_unknown} task(s) have no resolution verdict. Only the RESOLVED row")
    print( "      correlates with the leaderboard — the first two measure attempts, not")
    print( "      outcomes, and that gap is exactly failure F5. Wire up resolved_map.")
```

---

### PART 4: PACKAGING & LEADERBOARD SUBMISSION

#### Cell 12: `[Section 6 from Ryan's Notebook]` Package `submission.zip`
*Run unchanged from Ryan's notebook to create and validate `/kaggle/working/submission.zip`.*

---

#### Cell 13: `[RepoGraphForge]` Direct Download Link
```python
from IPython.display import FileLink
FileLink("submission.zip")
```
*Click the blue link to download `submission.zip` directly to your laptop, then upload it to the competition page to claim your daily submission slot!*
