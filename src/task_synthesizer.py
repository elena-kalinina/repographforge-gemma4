#!/usr/bin/env python3
"""
Milestone 3: Task Synthesizer via Reverse Fault Injection.
Prompts a teacher LLM (Gemma 4 31B via vLLM or Gemini) to introduce realistic,
subtle multi-node bugs across connected subgraph nodes, outputting paired
bug_patch, reproduction_test, problem_statement, and evidence_rubric.
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
inject a realistic, subtle multi-node bug (e.g. breaking an argument contract, subtle return type mismatch,
or state synchronization error between caller and callee).

Output strictly valid JSON matching this schema:
{
  "problem_statement": "Realistic GitHub issue report describing symptoms without naming internal functions.",
  "bug_patch": "Valid unified git diff that introduces the bug into the working code.",
  "reproduction_test": "A pytest test function that fails when bug_patch is applied and passes on clean code.",
  "evidence_rubric": [
    "Contract 1: Specifying parameter/type invariant...",
    "Contract 2: Specifying caller/callee contract...",
    "Contract 3: Specifying error/exception handling..."
  ]
}
"""


def build_mutation_prompt(subgraph: Dict[str, Any]) -> str:
    """Extracts node descriptions and source code from subgraph dictionary."""
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    nodes_summary = []
    snippets = []
    for n in nodes:
        nid = n.get("id", "")
        code = n.get("text", "").strip()
        nodes_summary.append(f"- `{nid}`")
        if code:
            # Truncate overly long functions to keep prompt compact
            code_preview = "\n".join(code.splitlines()[:60])
            snippets.append(f"### Node: {nid}\n```python\n{code_preview}\n```")

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

    for _ in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                parsed = safe_parse_json(content)
                if parsed and parsed.get("bug_patch") and parsed.get("problem_statement"):
                    return {
                        "synthetic_id": f"syn_{subgraph.get('graph_file', 'graph').split('.')[0]}_{subgraph.get('anchor_node', 'anchor')}",
                        "graph_file": subgraph.get("graph_file", ""),
                        "anchor_node": subgraph.get("anchor_node", ""),
                        "nodes_in_subgraph": [n.get("id") for n in subgraph.get("nodes", [])],
                        "problem_statement": parsed.get("problem_statement"),
                        "bug_patch": parsed.get("bug_patch"),
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

    # Checkpoint
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

    # Filter out already synthesized
    subgraphs_to_run = [
        s for s in subgraphs
        if f"syn_{s.get('graph_file', 'graph').split('.')[0]}_{s.get('anchor_node', 'anchor')}" not in already_done
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
