#!/usr/bin/env python3
"""
Milestone 3: Task Synthesizer via Reverse Fault Injection.
Prompts a teacher LLM (Gemma 4 31B via vLLM or Gemini) to introduce realistic,
subtle multi-node bugs across connected subgraph nodes, outputting paired:
  - problem_statement (without leaking exact node IDs)
  - target_node & target_file & target_line_span
  - buggy_snippet (the exact code present when the bug is active -> old_string for edit_file)
  - fixed_snippet (the clean original code -> new_string for edit_file)
  - bug_patch (diff introducing the bug)
  - gold_patch (inverted diff fixing the bug: Δ_gold = -Δ_bug)
  - evidence_rubric (structural caller/callee contracts)
"""

import argparse
import json
import logging
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("task_synthesizer")

SYNTHESIZER_SYSTEM_PROMPT = """You are an expert software engineering task synthesizer implementing RepoGraphForge.
Given a connected code-graph subgraph and the working Python source code of its nodes,
inject a realistic, subtle bug (e.g. breaking an argument contract, subtle return type mismatch,
missing edge-case guard, or caller/callee contract mismatch).

Output strictly valid JSON matching this schema:
{
  "problem_statement": "Realistic GitHub issue report describing the symptom/traceback without leaking internal node IDs.",
  "target_node": "Exact node ID modified to introduce the bug",
  "target_file": "Relative file path of the modified node (e.g. fastapi/routing.py)",
  "buggy_snippet": "The exact 3-12 lines of Python code AFTER injecting the bug (this is what the agent will see in the buggy repo).",
  "fixed_snippet": "The exact 3-12 lines of original clean Python code BEFORE the bug (the gold fix that restores working behavior).",
  "reproduction_test": "A pytest test function that fails on buggy_snippet and passes on fixed_snippet.",
  "evidence_rubric": [
    "Contract 1: Specifying parameter/type invariant...",
    "Contract 2: Specifying caller/callee contract...",
    "Contract 3: Specifying error/exception handling..."
  ]
}
"""


def infer_file_from_node(node: Dict[str, Any]) -> str:
    """Extracts or infers the relative Python file path for a graph node."""
    f = node.get("file") or node.get("file_path") or ""
    if f:
        return str(f)
    nid = str(node.get("id", "module"))
    parts = nid.split(".")
    if len(parts) >= 2:
        return "/".join(parts[:-1]) + ".py"
    return parts[0] + ".py"


def invert_unified_diff(diff_text: str) -> str:
    """Inverts a unified git diff (Δ_gold = -Δ_bug) by swapping +/- lines."""
    if not diff_text:
        return ""
    inverted = []
    for line in diff_text.splitlines():
        if line.startswith("--- a/"):
            inverted.append(line)
        elif line.startswith("+++ b/"):
            inverted.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            inverted.append("-" + line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            inverted.append("+" + line[1:])
        else:
            inverted.append(line)
    return "\n".join(inverted)


def build_unified_diff(target_file: str, old_code: str, new_code: str) -> str:
    """Constructs a clean unified git diff from old_code -> new_code."""
    old_lines = [f"-{l}" for l in old_code.splitlines()]
    new_lines = [f"+{l}" for l in new_code.splitlines()]
    body = "\n".join(old_lines + new_lines)
    return (
        f"diff --git a/{target_file} b/{target_file}\n"
        f"--- a/{target_file}\n"
        f"+++ b/{target_file}\n"
        f"@@ -1,{len(old_lines)} +1,{len(new_lines)} @@\n"
        f"{body}"
    )


def build_mutation_prompt(subgraph: Dict[str, Any]) -> str:
    """Extracts node descriptions and source code from subgraph dictionary."""
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    nodes_summary = []
    snippets = []
    for n in nodes:
        nid = n.get("id", "")
        fpath = infer_file_from_node(n)
        code = n.get("text", "").strip()
        nodes_summary.append(f"- `{nid}` (file: `{fpath}`)")
        if code:
            code_preview = "\n".join(code.splitlines()[:60])
            snippets.append(f"### Node: {nid} (file: {fpath})\n```python\n{code_preview}\n```")

    edges_summary = [
        f"- `{e.get('source')}` -> `{e.get('target')}` ({e.get('type', 'calls')})"
        for e in edges[:8]
    ]

    nodes_str = "\n".join(nodes_summary)
    edges_str = "\n".join(edges_summary) if edges_summary else "- Connected functional neighborhood"
    code_str = "\n\n".join(snippets[:5])

    return f"""### Subgraph Dependency Structure
Nodes:
{nodes_str}

Directed Contracts / Edges:
{edges_str}

### Clean Working Source Code
{code_str}
"""


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Cleans codeblock wrapping and fixes common JSON escaping issues."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text, strict=False)
    except Exception:
        pass
    try:
        fixed = re.sub(r'\\([^\'"\\/bfnrtu])', r'\\\\\1', text)
        return json.loads(fixed, strict=False)
    except Exception:
        return None


def synthesize_task_vllm(
    subgraph: Dict[str, Any],
    model_name: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    max_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """Synthesizes a single bug task from a subgraph using local vLLM."""
    prompt = build_mutation_prompt(subgraph)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    nodes = subgraph.get("nodes", [])
    anchor_node = subgraph.get("anchor_node", nodes[0].get("id", "anchor") if nodes else "anchor")
    default_file = infer_file_from_node(nodes[0]) if nodes else "app.py"

    for _ in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                parsed = safe_parse_json(content)
                if not parsed or not parsed.get("problem_statement"):
                    continue

                target_node = parsed.get("target_node") or anchor_node
                target_file = parsed.get("target_file") or default_file
                buggy_snippet = (parsed.get("buggy_snippet") or "").strip()
                fixed_snippet = (parsed.get("fixed_snippet") or "").strip()

                # Support both new schema (buggy_snippet/fixed_snippet) and legacy bug_patch
                raw_bug_patch = (parsed.get("bug_patch") or "").strip()
                if buggy_snippet and fixed_snippet and buggy_snippet != fixed_snippet:
                    bug_patch = build_unified_diff(target_file, fixed_snippet, buggy_snippet)
                    gold_patch = build_unified_diff(target_file, buggy_snippet, fixed_snippet)
                elif raw_bug_patch:
                    bug_patch = raw_bug_patch
                    gold_patch = invert_unified_diff(raw_bug_patch)
                else:
                    continue

                # Preserve compact subgraph metadata for grounded teacher rollout synthesis
                compact_nodes = []
                for n in nodes[:6]:
                    compact_nodes.append({
                        "id": n.get("id", ""),
                        "file": infer_file_from_node(n),
                        "start_line": n.get("start_line", n.get("line_start", 1)),
                        "end_line": n.get("end_line", n.get("line_end", 60)),
                        "text": (n.get("text") or "")[:1200],
                    })
                compact_edges = [
                    {"source": e.get("source"), "target": e.get("target"), "type": e.get("type", "calls")}
                    for e in subgraph.get("edges", [])[:8]
                ]

                g_name = subgraph.get("graph_file", subgraph.get("graph_snapshot", "graph")).split(".")[0]
                return {
                    "synthetic_id": f"syn_{g_name}_{anchor_node}",
                    "graph_file": subgraph.get("graph_file", subgraph.get("graph_snapshot", "")),
                    "anchor_node": anchor_node,
                    "target_node": target_node,
                    "target_file": target_file,
                    "buggy_snippet": buggy_snippet,
                    "fixed_snippet": fixed_snippet,
                    "nodes_in_subgraph": [n.get("id") for n in nodes],
                    "subgraph_nodes": compact_nodes,
                    "subgraph_edges": compact_edges,
                    "problem_statement": parsed.get("problem_statement"),
                    "bug_patch": bug_patch,
                    "fix_patch": gold_patch,
                    "gold_patch": gold_patch,
                    "reproduction_test": parsed.get("reproduction_test", ""),
                    "evidence_rubric": parsed.get("evidence_rubric", []),
                }
        except Exception:
            time.sleep(1)
    return None


def run_batch_synthesis(
    subgraphs_file: Path,
    output_file: Path,
    model_name: str,
    target_count: int = 400,
    max_workers: int = 4,
    base_url: str = "http://127.0.0.1:8000/v1",
) -> None:
    """Concurrent batch synthesis across sampled subgraphs."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    already_done = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    already_done.add(item.get("synthetic_id"))

    logger.info("Existing synthetic tasks: %d", len(already_done))

    subgraphs = []
    with subgraphs_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                subgraphs.append(json.loads(line))

    subgraphs_to_run = [
        s for s in subgraphs
        if f"syn_{s.get('graph_file', s.get('graph_snapshot', 'graph')).split('.')[0]}_{s.get('anchor_node', 'anchor')}" not in already_done
    ][:target_count]

    logger.info("Synthesizing %d new tasks using %d parallel workers on %s...", len(subgraphs_to_run), max_workers, model_name)

    start_time = time.time()
    success_count = len(already_done)

    with output_file.open("a", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(synthesize_task_vllm, s, model_name, base_url): s
            for s in subgraphs_to_run
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if res:
                out_f.write(json.dumps(res) + "\n")
                out_f.flush()
                success_count += 1
            if i % 25 == 0 or i == len(subgraphs_to_run):
                elapsed = time.time() - start_time
                logger.info("[%d/%d] Completed (Total valid tasks: %d, Elapsed: %.1fs)", i, len(subgraphs_to_run), success_count, elapsed)

    logger.info("Synthesis complete! Total tasks saved: %d -> %s", success_count, output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 3: Reverse task synthesizer.")
    parser.add_argument("--subgraphs-file", type=Path, default=Path("data/synthetic_tasks/sampled_subgraphs.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("data/synthetic_tasks/synthetic_tasks.jsonl"))
    parser.add_argument("--model-name", type=str, default="gemma-4-31b-it-qat-w4a16-ct")
    parser.add_argument("--target-count", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vllm-url", type=str, default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()

    run_batch_synthesis(
        subgraphs_file=args.subgraphs_file,
        output_file=args.output_file,
        model_name=args.model_name,
        target_count=args.target_count,
        max_workers=args.workers,
        base_url=args.vllm_url,
    )


if __name__ == "__main__":
    main()
