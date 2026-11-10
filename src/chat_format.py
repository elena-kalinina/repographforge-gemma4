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
CHAT_FORMAT_VERSION = "2026-11-10.stage1e"

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


def assert_trajectories_conform(records: List[Dict[str, Any]], *, sample: Optional[int] = None) -> None:
    """Gate 9: the DEMONSTRATIONS must use the same interface as the DECLARATIONS.

    `assert_tool_schema_verified` proves the declared schemas match the harness. It says
    nothing about the trajectories, and the two drifted independently: five of nine tools
    were mis-declared (F6) AND every trajectory supplied `submit_patch(description=...)`,
    an argument the harness does not define. Correcting one without the other leaves a
    model taught to ignore its own tool declarations.

    Cheap enough to run on every example; call it after loading and before training.
    """
    declared: Dict[str, set] = {}
    for t in COMPETITION_TOOLS:
        fn = t.get("function", t)
        declared[fn["name"]] = set((fn.get("parameters") or {}).get("properties", {}) or {})

    bad: List[str] = []
    seen = 0
    for i, rec in enumerate(records):
        if sample is not None and seen >= sample:
            break
        seen += 1
        for msg in rec.get("messages", []):
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", tc)
                name = fn.get("name")
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args, strict=False) if args.strip() else {}
                    except Exception:
                        bad.append(f"record {i}: {name} arguments unparseable")
                        continue
                if not isinstance(args, dict):
                    bad.append(f"record {i}: {name} arguments is {type(args).__name__}")
                    continue
                if name not in declared:
                    bad.append(f"record {i}: calls undeclared tool {name!r}")
                    continue
                extra = sorted(set(args) - declared[name])
                if extra:
                    bad.append(f"record {i}: {name} sends undeclared {extra}")
        if len(bad) >= 20:
            break

    if bad:
        raise RuntimeError(
            "\n" + "=" * 72
            + "\nTRAJECTORY GATE: TRAINING REFUSED\n"
            + "=" * 72
            + f"\n  {len(bad)} trajectory tool call(s) do not match COMPETITION_TOOLS.\n"
            + "".join(f"      {b}\n" for b in bad[:12])
            + "\n  The declared schemas and the demonstrated calls disagree. Verifying the\n"
              "  schemas (Cell 0C) does not check the data; run the Cell 8C rewrite pass in\n"
              "  the data preparation notebook, then re-run this.\n"
            + "=" * 72 + "\n"
        )
    logger.info("✓ Trajectory gate passed: %d record(s) use only declared parameters.", seen)


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
