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
│            *** BLOCKING — do not train until this passes ***           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│          PART 2: PHASE 4.1 SFT POST-TRAINING (NO RFT OVERWRITE)        │
│  [Cell 5] RepoGraphForge: GPU VRAM Flush & Health Check                │
│  [Cell 6] Write train_sft.py (Imports chat_format + Phase 1.2 Gate)    │
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
TOOL_RENDER_PROBES = ("old_string", "node_id", "submit_patch")

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

# Phase 3.4 Budget-Based System Prompt (replaces the brittle 1-file/turn-3 prohibitions)
DEFAULT_SYSTEM_PROMPT = """You are an autonomous software engineering agent equipped with a pre-computed repository code graph.

You have up to 15 tool calls. Spend them with discipline:
1. LOCALIZE VIA GRAPH FIRST:
   - Start with `get_code_subgraph`, `get_code_neighbors`, or `search_similar_code` on the symbol or symptom named in the issue/traceback.
   - If `get_code_subgraph` returns 0 nodes because the exact node ID is unknown, call `search_similar_code` to find the exact symbol and file path.
2. READ TARGETED SPANS (AT MOST 3 FILES):
   - Read at most 3 files using `read_file(filepath, start_line, end_line)` on the 30-80 lines around the target symbol.
   - Prefer inspecting the callee/helper function that the subgraph points to over paging through large caller files.
3. APPLY EXACT EDITS (`edit_file`):
   - Call `edit_file(filepath, old_string, new_string)` using the exact `old_string` copied verbatim from your `read_file` output.
   - If `edit_file` returns `FileEditError`, read the exact line range first and retry `edit_file`.
4. TEST & SUBMIT (`submit_patch`):
   - If you have tool calls remaining after `edit_file` succeeds, run the relevant unit test with `run_command` (e.g. `pytest tests/test_... -k ...`) to verify your fix.
   - Always finish with `submit_patch` once a valid edit is applied. Never call `submit_patch` before `edit_file` has succeeded."""

# Phase 3.2: Exact 9 Competition Tool Schemas declared in `agent.yaml`
COMPETITION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_code_subgraph",
            "description": "Extract the k-hop structural code dependency subgraph around a symbol or node_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Fully qualified symbol or node ID to query."},
                    "hops": {"type": "integer", "description": "Number of hops (default: 2)."},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_code_neighbors",
            "description": "Get immediate caller and callee neighbors of a symbol in the repository code graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Fully qualified symbol or node ID."},
                    "direction": {"type": "string", "enum": ["in", "out", "both"]},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_similar_code",
            "description": "Search the repository for symbols or functions semantically similar to the query using node embeddings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Symbol name, error message, or code concept to search."},
                    "top_k": {"type": "integer", "description": "Number of results to return (default: 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a specific line span of a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path to the file."},
                    "start_line": {"type": "integer", "description": "1-indexed starting line number."},
                    "end_line": {"type": "integer", "description": "1-indexed ending line number."},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string snippet (`old_string`) with `new_string` in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path to the file to edit."},
                    "old_string": {"type": "string", "description": "Exact existing lines to replace."},
                    "new_string": {"type": "string", "description": "New replacement lines."},
                },
                "required": ["filepath", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command (such as `pytest`) inside the repository sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Check current git status and modified files in the sandbox.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": "Finalize and submit the current git diff patch for evaluation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Brief summary of the fix."},
                },
            },
        },
    },
]


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

REQUIRED = ["STRICT_TOOLS", "TOOL_RENDER_PROBES", "MASK_PATH_COUNTS",
            "assert_tool_rendering_clean", "build_example", "verify_and_print_samples"]
missing = [n for n in REQUIRED if not hasattr(chat_format, n)]
assert not missing, f"STALE chat_format.py — missing {missing}. Re-run Cell 4."
assert chat_format.STRICT_TOOLS is True, "STRICT_TOOLS is off; tool-render failures will pass silently (F2)."
print("✓ chat_format.py is current; STRICT_TOOLS enabled.")
print(f"  tool render probes: {chat_format.TOOL_RENDER_PROBES}")
```

---

#### Cell 0C: `[Phase 0.2]` Diff Real ADK Tool Schemas Against `COMPETITION_TOOLS`

> **Placement note:** this is a Phase 0 diagnostic but it runs HERE, not in Part 0,
> because it needs three things defined by earlier cells: the ADK installed (Cell 1),
> Ryan's Section 3 tool definitions (Cell 3), and `/kaggle/working/chat_format.py`
> (Cell 4). Running it before those is what left it unexecuted.

> **Run this AFTER Ryan's Section 3 (which defines the tools) and BEFORE any training.**
> `src/chat_format.py:COMPETITION_TOOLS` is a hand-written transcription. Nothing has ever
> checked it against the real definitions. If a single parameter name differs, training
> teaches the policy to emit calls the sandbox rejects — which produces exactly the
> observed signature of a run that submits patches and scores zero.

```python
import json, inspect, sys
sys.path.insert(0, "/kaggle/working")
from chat_format import COMPETITION_TOOLS

# 1. Locate whatever Section 3 defined. Probe the usual names.
real_tools = None
for name in ("TOOLS", "tools", "TOOL_SCHEMAS", "AGENT_TOOLS", "FUNCTIONS", "tool_defs"):
    obj = globals().get(name)
    if isinstance(obj, (list, tuple)) and obj:
        real_tools = list(obj)
        print(f"✓ Found tool definitions in global `{name}` ({len(real_tools)} tools)")
        break

if real_tools is None:
    print("⚠️  Could not locate the tool definitions automatically.")
    print("    Candidate globals containing 'tool':")
    for k, v in sorted(globals().items()):
        if "tool" in k.lower() and not k.startswith("_"):
            print(f"      {k}: {type(v).__name__}")
    print("    Set `real_tools = <that object>` and re-run this cell.")
else:
    def normalize(t):
        """Reduce either a dict schema or a python callable to {name: set(param_names)}."""
        if isinstance(t, dict):
            fn = t.get("function", t)
            params = (fn.get("parameters") or {}).get("properties", {}) or {}
            return fn.get("name", "?"), set(params.keys())
        if callable(t):
            return t.__name__, set(inspect.signature(t).parameters.keys())
        decl = getattr(t, "declaration", None) or getattr(t, "_declaration", None)
        if decl is not None:
            params = getattr(decl, "parameters", None)
            props = getattr(params, "properties", {}) if params else {}
            return getattr(decl, "name", "?"), set(props.keys())
        return str(t), set()

    real = dict(normalize(t) for t in real_tools)
    ours = dict(normalize(t) for t in COMPETITION_TOOLS)

    print("\n" + "=" * 72)
    print("PHASE 0.2: TOOL SCHEMA DIFF  (real ADK  vs  COMPETITION_TOOLS)")
    print("=" * 72)

    mismatches = []
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
                print(f"  ✓ {name}({', '.join(sorted(real[name])) or ''})")

    if mismatches:
        print("\n  ✗ MISMATCHES FOUND:")
        for m in mismatches:
            print(f"      {m}")
        print("\n  >>> Do NOT train until these are reconciled. Prefer importing the real")
        print("      schemas over editing the copy — a copy drifts, an import cannot:")
        print("          import chat_format; chat_format.COMPETITION_TOOLS = real_tools")
    else:
        print("\n  ✓ All tool names and parameter names match. Schemas are safe to train on.")
    print("=" * 72)

    # 2. Confirm the declarations actually survive into rendered training text (failure F2).
    from pathlib import Path
    from transformers import AutoTokenizer
    from chat_format import render
    # Self-sufficient: do not depend on Cell 0A having run in this session.
    _MP = globals().get("MODEL_PATH") or Path(
        "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"
    )
    _tok = AutoTokenizer.from_pretrained(str(_MP), local_files_only=True)
    try:
        _probe = render([{"role": "user", "content": "probe"}], _tok, tools=COMPETITION_TOOLS)
    except RuntimeError as e:
        print(f"\n>>> F2 IS LIVE: {e}")
        raise SystemExit(0)
    _n = len(_tok(_probe, add_special_tokens=False)["input_ids"])
    print(f"\nRendered preamble with tools: {_n} tokens")
    print(f"Declarations present in text: {all(p in _probe for p in ('old_string','node_id','submit_patch'))}")
    print("If that is False, the template is ignoring `tools=` and failure F2 is live.")
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
